import argparse
import csv
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import jaccard_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from test_tube import HyperOptArgumentParser
import seaborn as sn
import matplotlib
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from transformers import get_linear_schedule_with_warmup

from dataloader import sentiment_analysis_dataset, MyCollator, EMOTION_COLS

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging  # Add this

log = logging.getLogger(__name__)  #
from RedditTransformer import RedditTransformer


class BERTRegressor(pl.LightningModule):

    def __init__(self, hparams) -> None:
        super(BERTRegressor, self).__init__()

        # --- THE FIX: Clean the hparams so TensorBoard doesn't crash ---
        # 1. Convert the test_tube namespace into a standard dictionary
        if hasattr(hparams, '__dict__'):
            hparams_dict = vars(hparams)
        else:
            hparams_dict = dict(hparams)

        # 2. Filter out all the internal test_tube junk. Keep ONLY basic data.
        clean_hparams = {
            k: v for k, v in hparams_dict.items()
            if isinstance(v, (int, float, str, bool, type(None)))
        }

        # 3. Save the squeaky-clean version
        self.save_hyperparameters(clean_hparams)
        # ---------------------------------------------------------------

        # PL 2.0 FIX: Initialize lists to hold step outputs
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # Use self.hparams which is now safely managed by Lightning
        self.batch_size = self.hparams.batch_size

        # Initialize the collator
        self.prepare_sample = MyCollator(self.hparams.encoder_model, self.hparams.max_length)

        self.__build_model()
        self.__build_loss()
        if self.hparams.nr_frozen_epochs > 0:
            self.freeze_encoder()
        else:
            self._frozen = False

        self.nr_frozen_epochs = self.hparams.nr_frozen_epochs
    def forward(self, batch: dict) -> dict:
        """
        The front door of the PyTorch Lightning module.
        Takes the raw batch, passes it to the Transformer, and packages the results.
        """
        # 1. Send the data to your custom RedditTransformer
        logits_main, logits_aux, hidden_states = self.model(batch)

        # 2. Package the tuple into the exact dictionary the rest of your code expects!
        return {
            "logits": logits_main,
            "logits_aux": logits_aux,
            "hidden_states": hidden_states
        }

    def predict(self, sample: dict) -> dict:
        """ Predict function.
        :param sample: dictionary with the text we want to classify.

        Returns:
            Dictionary with the input text and the predicted label.
        """
        if self.training:
            self.eval()

        with torch.no_grad():
            model_input = self.prepare_sample([sample])
            model_out = self.forward(model_input)
            logits = model_out["logits"].numpy()
            predicted_labels = [
                self.label_encoder.index_to_token[prediction]
                for prediction in np.argmax(logits, axis=1)
            ]
            sample["predicted_label"] = predicted_labels[0]

        return sample

    def __build_model(self):
        try:
            train_df = pd.read_csv(self.hparams.train_csv)
            test_df = pd.read_csv(self.hparams.test_csv)
            dev_df = pd.read_csv(self.hparams.dev_csv)
            comments = pd.concat([train_df, test_df, dev_df])
        except Exception as e:
            print(f"Could not load csv check for correct configurations path {e}")

        self.hparams.le = LabelEncoder()
        self.hparams.le_aux = LabelEncoder()

        aux_task_str = str(self.hparams.aux_task)
        if aux_task_str not in ('None', 'emotions'):
            self.hparams.le_aux.fit(comments[self.hparams.aux_task].values)
            self.model = RedditTransformer(self.hparams.encoder_model, 1, self.hparams.extra_dropout,
                                           len(self.hparams.le_aux.classes_))
            self.weights = nn.Parameter(
                torch.Tensor([1 + self.hparams.loss_aux_dropout, 1 - self.hparams.loss_aux_dropout]),
                requires_grad=True)
            self.alpha = 0.5
        elif aux_task_str == 'emotions':
            # Build model with only the learnable emotion classes (defined in EMOTION_COLS)
            self.model = RedditTransformer(self.hparams.encoder_model, 1, self.hparams.extra_dropout, len(EMOTION_COLS))
            # For best emotion prediction, give the AUX task the larger initial weight.
            # weights[0] = main task (usVSthem regression)  = 1 - loss_aux_dropout
            # weights[1] = aux  task (emotion Jaccard)      = 1 + loss_aux_dropout
            # With loss_aux_dropout=0.25 → main=0.75, aux=1.25
            # With loss_aux_dropout=0.85 → main=0.15, aux=1.85 (heavy emotion focus)
            self.weights = nn.Parameter(
                torch.Tensor([1 - self.hparams.loss_aux_dropout, 1 + self.hparams.loss_aux_dropout]),
                requires_grad=True)
            self.alpha = 0.5
        else:
            # reset label encoder
            self.hparams.le_aux = LabelEncoder()
            self.model = RedditTransformer(self.hparams.encoder_model, 1, self.hparams.extra_dropout, None)

    def __build_loss(self):  # FIX: was build_loss — missing __ prefix
        self._loss = nn.MSELoss()

        if self.hparams.aux_task == 'emotions':
            # Compute class-balanced pos_weight from training data.
            # pos_weight[i] = neg_count[i] / pos_count[i]
            # This forces the model to put proportionally MORE penalty on
            # missing rare emotions (e.g., Relief: 20 samples → weight≈183)
            # vs common ones (e.g., Contempt: 1396 samples → weight≈1.6).
            # Without this, the model learns to predict 0 for rare classes
            # because that's the safest path to a low average BCE loss.
            train_df   = pd.read_csv(self.hparams.train_csv)
            pos_counts = train_df[EMOTION_COLS].sum().values.astype(float)   # (N_emotions,)
            neg_counts = (len(train_df) - pos_counts).astype(float)
            # Clamp min to 1 to avoid div-by-zero for any class with 0 positives
            pos_weight = torch.tensor(
                neg_counts / np.maximum(pos_counts, 1.0), dtype=torch.float32
            )
            log.info(f"BCEWithLogitsLoss pos_weight: {pos_weight.tolist()}")
            self._loss_aux = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self._loss_aux = nn.CrossEntropyLoss()  # FIX: was self.loss_aux — missing _ prefix

        # abs loss
        self._gradLoss = nn.L1Loss()  # FIX: was self.gradLoss — missing _ prefix

    def unfreeze_encoder(self) -> None:
        """ un-freezes the encoder layer. """
        if self._frozen:
            log.info(f"\n-- Encoder model fine-tuning")
            for param in self.model.encoder.parameters():
                param.requires_grad = True
            self._frozen = False

    def freeze_encoder(self) -> None:
        """ freezes the encoder layer. """
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        for param in self.model.encoder.encoder.layer[-1].output.parameters():
            param.requires_grad = True
        self._frozen = True

    def loss(self, predictions: dict, targets: dict) -> torch.Tensor:

        loss = self._loss(predictions['logits'].flatten(), targets['labels'])
        # Consistent multi-task check (handles both None object and 'None' string)
        aux_active = str(self.hparams.aux_task) != "None"

        if self.hparams.aux_task == 'emotions':
            loss_aux = self._loss_aux(predictions["logits_aux"], targets["labels_aux"])
            return loss, loss_aux
        elif aux_active:
            loss_aux = self._loss_aux(predictions["logits_aux"], targets["labels_aux"].type(torch.long))
            return loss, loss_aux
        return loss, None

    def backward(self, loss, *args, **kwargs):
        aux_active = str(self.hparams.aux_task) != "None"
        if aux_active and self.hparams.gradnorm == True:
            # Compute how much the main task loss and the auxiliary loss weigh in the final loss
            loss_val = self.weights * loss
            total_weighted_loss = loss_val.sum()
            # back propagate the loss and remember the gradients
            total_weighted_loss.backward(retain_graph=True)
            # reset the loses
            self.weights.grad.data.zero_()
            # FIX: was self.model.encoder.layer[-1] — missing the inner .encoder,
            W = list(self.model.encoder.encoder.layer[-1].output.parameters())
            norms = []

            for w_i, L_i in zip(self.weights, loss.flatten()):
                # gradient of L_i(t) w.r.t. W the last layer
                gLgW = torch.autograd.grad(L_i, W, retain_graph=True)

                # Sum of squared norms of all gradients
                norm = 0
                # sum up the norm of all 8 matrices this is the l2 score
                for g in gLgW:
                    norm += torch.norm(g) ** 2
                norm = torch.sqrt(norm)
                norms.append(norm * w_i)

            # FIX: norms was a plain Python list — lists have no .mean(), needs torch.stack first
            norms = torch.stack(norms)

            # take the losses at the beginning of the batch processing
            if getattr(self, "initial_loses", None) is None or self.trainer.global_step == 0:
                self.initial_loses = loss.detach()

            with torch.no_grad():
                # compute the loss differences
                loss_ratios = loss / self.initial_loses
                # compute loss of task 1 and 2 compared to their average
                inverse_train_rates = loss_ratios / loss_ratios.mean()

                # use the average of the means to compute the new target for the force of the task
                constant_term = (norms.mean() * (inverse_train_rates ** self.alpha)).detach()

            # Output - Target take its derivative and modify weight main task and auxiliary task
            grad_norm_loss = (norms - constant_term).abs().sum()
            # update the weights as the loss is wnew = w old - w * norms * sign * lr
            self.weights.grad = torch.autograd.grad(grad_norm_loss, self.weights)[0]
        elif aux_active:
            # Grad norm is not open so just try to minimise the error in the fastest way possible
            loss_val = self.weights * loss
            total_weighted_loss = loss_val.sum()
            total_weighted_loss.backward()
        else:
            loss.backward()

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure, **kwargs):
        # Phase 1: Execute the closure and update weights
        optimizer.step(closure=optimizer_closure)

        # Phase 2: The Multi-Task Safety Net
        if str(getattr(self.hparams, 'aux_task', 'None')) != 'None':
            with torch.no_grad():
                # Prevent negative weights which invert loss
                self.weights.data = torch.clamp(self.weights.data, min=1e-4)
                # Keep the normalization so weights always add up to 2
                normalize_coeff = len(self.weights) / self.weights.sum()
                self.weights.data = self.weights.data * normalize_coeff

    def _compute_aux_metrics_categorical(
            self,
            y_aux: torch.Tensor,
            y_aux_hat: torch.Tensor,
            device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-batch aux accuracy for non-emotions categorical aux tasks."""
        label_hat_aux = torch.argmax(y_aux_hat, dim=1)
        acc = (y_aux == label_hat_aux).float().mean()
        cm = confusion_matrix(
            self.hparams.le_aux.inverse_transform(y_aux.long().cpu().numpy()),
            self.hparams.le_aux.inverse_transform(label_hat_aux.cpu().numpy()),
            labels=self.hparams.le_aux.classes_,
        )
        return (
            torch.tensor(float(acc), device=device),
            torch.tensor(cm, device=device),
        )

    def _compute_jaccard_epoch(
            self,
            all_labels_aux: torch.Tensor,   # (N, 13) float, values 0 or 1
            all_logits_aux: torch.Tensor,   # (N, 13) raw logits
            dataset_columns,
            device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Jaccard and confusion matrix over the FULL epoch for emotions.
        Must be called once at epoch-end after concatenating all batches.
        Per-batch Jaccard is wrong: sparse batches make sklearn set per-class
        score to 0.0 for labels with no true/predicted samples, collapsing
        the macro average to near-zero even for a well-trained model.
        """
        labels_hat = (torch.sigmoid(all_logits_aux) > 0.5)  # (N, 13) bool
        y_np   = all_labels_aux.cpu().numpy().astype(int)   # (N, 13)
        hat_np = labels_hat.cpu().numpy().astype(int)       # (N, 13)

        # zero_division=0 suppresses the warning; classes absent from the
        # entire split are correctly scored as 0 (rare but valid).
        acc = jaccard_score(y_np, hat_np, average="macro", zero_division=0)

        cm = confusion_matrix(
            y_np.argmax(axis=1),
            hat_np.argmax(axis=1),
            labels=np.arange(len(dataset_columns)),
        )
        return (
            torch.tensor(float(acc), device=device),
            torch.tensor(cm, device=device),
        )

    def validation_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        inputs, targets = batch
        model_out = self.forward(inputs)
        loss_val = self.loss(model_out, targets)

        y     = targets["labels"]
        y_hat = model_out["logits"]
        device = loss_val[0].device

        aux_task_str = str(self.hparams.aux_task)
        if aux_task_str == "emotions":
            # Store raw tensors — Jaccard will be computed over the FULL epoch
            output = {
                "val_loss":       loss_val[0],
                "labels":         y,
                "predictions":    y_hat,
                "labels_aux":     targets["labels_aux"],    # (B, 13) float
                "logits_aux":     model_out["logits_aux"],  # (B, 13) raw
            }
        elif aux_task_str != "None":
            # Categorical aux: per-batch accuracy is fine
            val_acc_aux, conf_matrix_aux = self._compute_aux_metrics_categorical(
                targets["labels_aux"],
                model_out["logits_aux"],
                device=device,
            )
            output = {
                "val_loss":        loss_val[0],
                "labels":          y,
                "predictions":     y_hat,
                "conf_matrix_aux": conf_matrix_aux,
                "val_acc_aux":     val_acc_aux,
            }
        else:
            output = {
                "val_loss":        loss_val[0],
                "labels":          y,
                "predictions":     y_hat,
                "conf_matrix_aux": torch.zeros(1, 1, device=device),
                "val_acc_aux":     torch.zeros(1, device=device),
            }

        self.validation_step_outputs.append(output)
        return output

    def training_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        inputs, targets = batch
        model_out = self.forward(inputs)
        loss_val = self.loss(model_out, targets)

        if str(self.hparams.aux_task) != "None":
            task_losses = torch.stack(loss_val)
            total_weighted_loss = (self.weights * task_losses).sum()
            log_dict = {
                "train_loss": total_weighted_loss,
                "weight1": self.weights[0],
                "weight2": self.weights[1],
            }
        else:
            task_losses = loss_val[0]
            total_weighted_loss = task_losses
            log_dict = {"train_loss": total_weighted_loss}

        # Modern Lightning: self.log() instead of returning dicts
        for k, v in log_dict.items():
            self.log(k, v, prog_bar=True, sync_dist=True)

        output = {"loss": task_losses}
        # PL 2.0 FIX: Store step output
        self.training_step_outputs.append(output)
        return output

    def test_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        inputs, targets = batch
        model_out = self.forward(inputs)
        loss_val = self.loss(model_out, targets)

        y     = targets["labels"]
        y_hat = model_out["logits"]
        device = loss_val[0].device

        aux_task_str = str(self.hparams.aux_task)
        if aux_task_str == "emotions":
            # Store raw tensors — Jaccard computed over full epoch
            output = {
                "val_loss":    loss_val[0],
                "labels":      y,
                "predictions": y_hat,
                "labels_aux":  targets["labels_aux"],
                "logits_aux":  model_out["logits_aux"],
            }
        elif aux_task_str != "None":
            val_acc_aux, conf_matrix_aux = self._compute_aux_metrics_categorical(
                targets["labels_aux"],
                model_out["logits_aux"],
                device=device,
            )
            output = {
                "val_loss":        loss_val[0],
                "labels":          y,
                "predictions":     y_hat,
                "conf_matrix_aux": conf_matrix_aux,
                "val_acc_aux":     val_acc_aux,
            }
        else:
            output = {
                "val_loss":        loss_val[0],
                "labels":          y,
                "predictions":     y_hat,
                "conf_matrix_aux": torch.zeros(1, 1, device=device),
                "val_acc_aux":     torch.zeros(1, device=device),
            }

        self.test_step_outputs.append(output)
        return output

    # PL 2.0 FIX: Renamed train_epoch_end to on_train_epoch_end and removed outputs arg
    def on_train_epoch_end(self) -> None:
        """
        Calculates the average training loss across the entire epoch.
        """
        outputs = self.training_step_outputs

        # FIX: was using old progress_bar/log return dict pattern — replaced with self.log()
        # torch.stack + .mean() replaces the manual loop and the old DP reduction hack
        train_loss_mean = torch.stack([o["loss"] for o in outputs]).mean()

        # Note: Look for 'loss', not 'val_loss', because this is the training step!
        self.log("train_loss", train_loss_mean, prog_bar=True, sync_dist=True)

        # PL 2.0 FIX: Clear memory
        self.training_step_outputs.clear()

    # PL 2.0 FIX: Renamed validation_epoch_end to on_validation_epoch_end and removed outputs arg
    def on_validation_epoch_end(self) -> None:
        """
        Aggregates validation step outputs and logs metrics.
        Modern Lightning (>=1.6) handles device/DDP reduction automatically.
        """
        outputs = self.validation_step_outputs
        aux_task_str = str(self.hparams.aux_task)

        # --- Loss ---
        val_loss_mean = torch.stack([o["val_loss"] for o in outputs]).mean()

        # --- Main task: concatenate predictions and labels ---
        val_y     = torch.cat([o["labels"]      for o in outputs])  # (N,)
        val_y_hat = torch.cat([o["predictions"] for o in outputs])  # (N,)

        # --- Pearson correlation (torch-native) ---
        def _pearson(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x = x.float().flatten()
            y = y.float().flatten()
            x_c = x - x.mean()
            y_c = y - y.mean()
            return (x_c * y_c).sum() / (x_c.norm() * y_c.norm() + 1e-8)

        pearsonr = _pearson(val_y, val_y_hat)

        # --- Aux metrics ---
        device = val_loss_mean.device
        if aux_task_str == "emotions":
            # Jaccard computed over FULL epoch to avoid ill-defined per-batch scores
            all_labels_aux = torch.cat([o["labels_aux"] for o in outputs])   # (N, 13)
            all_logits_aux = torch.cat([o["logits_aux"] for o in outputs])   # (N, 13)
            val_acc_aux, conf_matrix_aux = self._compute_jaccard_epoch(
                all_labels_aux, all_logits_aux,
                dataset_columns=self._dev_dataset.columns,
                device=device,
            )
        elif aux_task_str != "None":
            val_acc_aux    = torch.stack([o["val_acc_aux"]     for o in outputs]).mean()
            conf_matrix_aux = torch.stack([o["conf_matrix_aux"] for o in outputs]).sum(dim=0)
        else:
            val_acc_aux     = torch.zeros(1, device=device)
            conf_matrix_aux = torch.zeros(1, 1, device=device)

        # --- Log metrics ---
        self.log("val_loss",    val_loss_mean, prog_bar=True, sync_dist=True)
        self.log("val_pearson", pearsonr,      prog_bar=True, sync_dist=True)
        self.log("val_acc_aux", val_acc_aux,   prog_bar=True, sync_dist=True)

        # --- Confusion matrix figure (rank 0 only) ---
        if self.trainer.is_global_zero and aux_task_str != "None":
            labels = (
                self._dev_dataset.columns
                if aux_task_str == "emotions"
                else self.hparams.le_aux.classes_
            )
            fig, ax = plt.subplots(figsize=(10, 7))
            sn.heatmap(conf_matrix_aux.float().cpu(), annot=True, ax=ax)
            ax.set_xlabel("Predicted labels")
            ax.set_ylabel("True labels")
            ax.set_title("Confusion Matrix")
            ax.xaxis.set_ticklabels(labels)
            ax.yaxis.set_ticklabels(labels)
            self.logger.experiment.add_figure("confusion matrix Aux", fig)
            plt.close(fig)

        # PL 2.0 FIX: Clear memory
        self.validation_step_outputs.clear()

    # PL 2.0 FIX: Renamed test_epoch_end to on_test_epoch_end and removed outputs arg
    def on_test_epoch_end(self) -> None:
        """Aggregates test step outputs and logs metrics."""
        outputs = self.test_step_outputs
        aux_task_str = str(self.hparams.aux_task)

        val_loss_mean = torch.stack([o["val_loss"] for o in outputs]).mean()

        val_y     = torch.cat([o["labels"]      for o in outputs])
        val_y_hat = torch.cat([o["predictions"] for o in outputs])

        def _pearson(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x = x.float().flatten()
            y = y.float().flatten()
            x_c = x - x.mean()
            y_c = y - y.mean()
            return (x_c * y_c).sum() / (x_c.norm() * y_c.norm() + 1e-8)

        pearsonr = _pearson(val_y, val_y_hat)

        # --- Aux metrics ---
        device = val_loss_mean.device
        if aux_task_str == "emotions":
            # Jaccard over FULL test set
            all_labels_aux = torch.cat([o["labels_aux"] for o in outputs])
            all_logits_aux = torch.cat([o["logits_aux"] for o in outputs])
            val_acc_aux, conf_matrix_aux = self._compute_jaccard_epoch(
                all_labels_aux, all_logits_aux,
                dataset_columns=self._test_dataset.columns,
                device=device,
            )
        elif aux_task_str != "None":
            val_acc_aux     = torch.stack([o["val_acc_aux"]     for o in outputs]).mean()
            conf_matrix_aux = torch.stack([o["conf_matrix_aux"] for o in outputs]).sum(dim=0)
        else:
            val_acc_aux     = torch.zeros(1, device=device)
            conf_matrix_aux = torch.zeros(1, 1, device=device)

        self.log("test_loss",    val_loss_mean, prog_bar=True, sync_dist=True)
        self.log("test_pearson", pearsonr,      prog_bar=True, sync_dist=True)
        self.log("test_acc_aux", val_acc_aux,   prog_bar=True, sync_dist=True)

        # Save predictions to CSV
        val_labels = val_y.float().cpu().numpy().flatten()
        val_preds  = val_y_hat.float().cpu().numpy().flatten()
        pred_path  = Path(self.hparams.checkpoint_path) / "predictions.csv"
        with pred_path.open("w", newline="") as f:
            csv.writer(f).writerows(zip(val_preds, val_labels))

        if self.trainer.is_global_zero and aux_task_str != "None":
            labels = (
                self._test_dataset.columns
                if aux_task_str == "emotions"
                else self.hparams.le_aux.classes_
            )
            fig, ax = plt.subplots(figsize=(10, 7))
            sn.heatmap(conf_matrix_aux.float().cpu(), annot=True, ax=ax)
            ax.set_xlabel("Predicted labels")
            ax.set_ylabel("True labels")
            ax.set_title("Confusion Matrix")
            ax.xaxis.set_ticklabels(labels)
            ax.yaxis.set_ticklabels(labels)
            self.logger.experiment.add_figure("confusion matrix Aux", fig)
            plt.close(fig)

        # PL 2.0 FIX: Clear memory
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        """Sets different learning rates for different parameter groups."""
        aux_task_str = str(self.hparams.aux_task)

        if aux_task_str != "None" and self.hparams.gradnorm:
            param_groups = [
                # FIX: Changed learning_rate to encoder_learning_rate
                {"params": self.model.parameters(), "lr": self.hparams.encoder_learning_rate},
                {"params": self.weights, "lr": 1e-2},
            ]
        elif self.hparams.aux_task == "emotions":
            # Match the ACTUAL parameter names in RedditTransformer:
            #   BertEncoder  → self.layer_aux  (the split 12th transformer block)
            #   BertPooler   → self.dense_aux  (the split CLS pooler head)
            #   BertRegressor → self.classification_head_aux (the output MLP)
            # Bug was: "dense_emotions" and "layer_emotion" — those names don't exist!
            aux_keywords = {"classification_head_aux", "dense_aux", "layer_aux"}
            params, aux_params = [], []
            for name, param in self.model.named_parameters():
                if any(kw in name for kw in aux_keywords):
                    aux_params.append(param)
                else:
                    params.append(param)
            param_groups = [
                {"params": params,     "lr": self.hparams.encoder_learning_rate},
                # Reduced from 10x → 3x: the emotion head still learns faster than
                # the shared BERT backbone, but 10x was causing rapid memorization
                # of the small training set (3,683 examples) and severe overfitting.
                {"params": aux_params, "lr": self.hparams.encoder_learning_rate * 3},
            ]
        else:
            param_groups = [
                # FIX: Changed learning_rate to encoder_learning_rate
                {"params": self.model.parameters(), "lr": self.hparams.encoder_learning_rate},
            ]

        optimizer = optim.Adam(param_groups)

        train_steps = len(self.train_dataloader()) * self.hparams.max_epochs
        warmup_steps = int(self.hparams.warmup_proportion * train_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=train_steps,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def on_epoch_end(self):
        """ Pytorch lightning hook """
        if self.current_epoch + 1 >= self.nr_frozen_epochs:
            self.unfreeze_encoder()

    def __retrieve_dataset(self, train=True, val=True, test=True):
        """ Retrieves task specific dataset """
        return sentiment_analysis_dataset(self.hparams, train, val, test)

    def train_dataloader(self) -> DataLoader:  # FIX: removed @pl.data_loader — deprecated and removed in Lightning 1.0
        """ Function that loads the train set. """
        self._train_dataset = self.__retrieve_dataset(val=False, test=False)
        return DataLoader(
            dataset=self._train_dataset,
            sampler=RandomSampler(self._train_dataset),
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    def val_dataloader(self) -> DataLoader:  # FIX: removed @pl.data_loader
        """ Function that loads the validation set. """
        self._dev_dataset = self.__retrieve_dataset(train=False, test=False)
        return DataLoader(
            dataset=self._dev_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    def test_dataloader(self) -> DataLoader:  # FIX: removed @pl.data_loader
        """ Function that loads the test set. """
        self._test_dataset = self.__retrieve_dataset(train=False, val=False)
        return DataLoader(
            dataset=self._test_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    @classmethod
    def add_model_specific_args(cls, parser: HyperOptArgumentParser) -> HyperOptArgumentParser:

        parser.add_argument(
            "--encoder_model",
            default="bert-base-uncased",
            type=str,
            help="Encoder model to use",
        )
        parser.add_argument(
            "--gradnorm",
            default=False,
            type=bool,
            help=" This is used by the optimizer to balance the weight of a task"
        )
        parser.opt_list(
            "--aux_task",
            default=None,
            tunable=False,
            type=str,
            options=[None, 'bias', 'emotions'],
            help="The name of the task to train",
        )
        parser.add_argument(
            "--encoder_learning_rate",
            default=1e-05,
            type=float,
            help="The learning rate for the encoder model",
        )
        parser.opt_list(
            "--loss_aux_dropout",
            default=0.25,
            type=float,
            options=[-0.85, -0.25, 0.25, 0.85],
            help="Add dropout to Transformer one task is 1-alpha the other 1 + alpha.",
        )
        parser.opt_list(
            "--warmup_aux",
            default=5,
            tunable=True,
            options=[3, 5, 7, 10],
            type=int,
            help="Add warmup scheduled learning this means that the model uses also the auxiliary task for the first epochs when trying to predict only the first task.",
        )
        parser.opt_list(
            "--extra_dropout",
            default=0,
            tunable=False,
            options=[0, 0.05, 0.1, 0.15, 0.2],
            type=float,
            help="Add dropout to Transformer .",
        )
        parser.opt_list(
            "--warmup_proportion",
            default=0,
            tunable=False,
            options=[0, 0.1, 0.2, 0.3],
            type=float,
            help="Add warmup to Transformer.",
        )
        parser.opt_list(
            "--nr_frozen_epochs",
            default=0,
            type=int,
            help="Number of epochs we want to keep the encoder model frozen.",
            tunable=False,
            options=[0, 1, 2, 3, 4, 5],
        )
        parser.add_argument(
            "--max_length",
            default=512,
            type=int,
            help="Max length for text.",
        )
        # Data Args:
        parser.add_argument(
            "--label_set",
            default="pos,neg",
            type=str,
            help="Classification labels set.",
        )
        parser.add_argument(
            "--train_csv",
            default="Resources/UsVsThem_train_public.csv",
            type=str,
            help="Path to the file containing the train data.",
        )
        parser.add_argument(
            "--dev_csv",
            default="Resources/UsVsThem_valid_public.csv",
            type=str,
            help="Path to the file containing the train data.",
        )
        parser.add_argument(
            "--test_csv",
            default="Resources/UsVsThem_test_public.csv",
            type=str,
            help="Path to the file containing the train data.",
        )
        parser.add_argument(
            "--loader_workers",
            default=8,
            type=int,
            help="How many subprocesses to use for data loading. 0 means that \
                   the data will be loaded in the main process.",
        )
        return parser