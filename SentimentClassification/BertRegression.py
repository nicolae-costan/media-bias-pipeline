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

from dataloader import sentiment_analysis_dataset, MyCollator

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
            # we have 13 emotions
            self.model = RedditTransformer(self.hparams.encoder_model, 1, self.hparams.extra_dropout, 13)
            self.weights = nn.Parameter(
                torch.Tensor([1 + self.hparams.loss_aux_dropout, 1 - self.hparams.loss_aux_dropout]),
                requires_grad=True)
            self.alpha = 0.5
        else:
            # reset label encoder
            self.hparams.le_aux = LabelEncoder()
            self.model = RedditTransformer(self.hparams.encoder_model, 1, self.hparams.extra_dropout, None)

    def __build_loss(self):  # FIX: was build_loss — missing __ prefix
        self._loss = nn.MSELoss()

        if self.hparams.aux_task == 'emotions':
            self._loss_aux = nn.BCEWithLogitsLoss()  # FIX: was self.loss_aux — missing _ prefix
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
        # FIX: was (aux_task != 'None') & (aux_task == 'emotions') — logically impossible,
        # 'emotions' is never equal to 'None' so the & made the branch unreachable
        if self.hparams.aux_task == 'emotions':
            loss_aux = self._loss_aux(predictions["logits_aux"], targets["labels_aux"])
            return loss, loss_aux
        elif self.hparams.aux_task != 'None':
            loss_aux = self._loss_aux(predictions["logits_aux"], targets["labels_aux"].type(torch.long))
            return loss, loss_aux
        return loss, None

    def backward(self, loss, *args, **kwargs):
        if self.hparams.aux_task != 'None' and self.hparams.gradnorm == True:
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
            if self.trainer.global_step == 0:
                self.initial_loses = loss.detach()

            with torch.no_grad():
                # compute the loss differences
                loss_ratios = loss / self.initial_loses
                # compute loss of task 1 and 2 compared to their average
                inverse_train_rates = loss_ratios / loss_ratios.mean()

                # use the average of the means to compute the new target for the force of the task
                constant_term = norms.mean() * (inverse_train_rates ** self.alpha)

            # Output - Target take its derivative and modify weight main task and auxiliary task
            grad_norm_loss = (norms - constant_term).abs().sum()
            # update the weights as the loss is wnew = w old - w * norms * sign * lr
            self.weights.grad = torch.autograd.grad(grad_norm_loss, self.weights)[0]
        elif self.hparams.aux_task != 'None':
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
        if getattr(self.hparams, 'aux_task', 'None') != 'None':
            with torch.no_grad():
                # Keep the normalization so weights always add up to 2
                normalize_coeff = len(self.weights) / self.weights.sum()
                self.weights.data = self.weights.data * normalize_coeff

    def _compute_aux_metrics(
            self,
            y_aux: torch.Tensor,
            y_aux_hat: torch.Tensor,
            device: torch.device,
            dataset_columns=None,  # FIX: added parameter so caller can pass the right dataset columns
            # (val uses _dev_dataset, test uses _test_dataset)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared aux metric logic for val/test steps. Stays on-device where possible."""

        if self.hparams.aux_task not in ("None", "emotions"):
            label_hat_aux = torch.argmax(y_aux_hat, dim=1)
            acc = (y_aux == label_hat_aux).float().mean()

            # sklearn needs CPU — unavoidable, but isolated here
            cm = confusion_matrix(
                self.hparams.le_aux.inverse_transform(y_aux.long().cpu().numpy()),
                self.hparams.le_aux.inverse_transform(label_hat_aux.cpu().numpy()),
                labels=self.hparams.le_aux.classes_,
            )

        elif self.hparams.aux_task == "emotions":
            labels_hat = (torch.sigmoid(y_aux_hat) > 0.5)
            # jaccard_score requires CPU — unavoidable
            acc = jaccard_score(y_aux.cpu().numpy(), labels_hat.cpu().numpy(), average="macro")
            acc = float(acc)

            # FIX: was hardcoded to self._dev_dataset.columns — now uses the passed-in columns
            # so test_step can correctly pass self._test_dataset.columns
            cm = confusion_matrix(
                y_aux.cpu().numpy().argmax(axis=1),
                labels_hat.cpu().numpy().argmax(axis=1),
                labels=np.arange(len(dataset_columns)),
            )

        else:
            return (
                torch.zeros(1, device=device),
                torch.zeros(1, 1, device=device),
            )

        return (
            torch.tensor(acc, device=device),
            torch.tensor(cm, device=device),
        )

    def validation_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        inputs, targets = batch
        model_out = self.forward(inputs)
        loss_val = self.loss(model_out, targets)

        y = targets["labels"]
        y_hat = model_out["logits"]

        if self.hparams.aux_task != "None":
            val_acc_aux, conf_matrix_aux = self._compute_aux_metrics(
                targets["labels_aux"],
                model_out["logits_aux"],
                device=loss_val[0].device,  # FIX: was loss_val.device — loss() returns a tuple, not a tensor
                dataset_columns=self._dev_dataset.columns,  # FIX: pass correct columns for val
            )
        else:
            val_acc_aux = torch.zeros(1, device=loss_val[0].device)  # FIX: same tuple indexing fix
            conf_matrix_aux = torch.zeros(1, 1, device=loss_val[0].device)

        output = {
            "val_loss": loss_val[0],  # FIX: unpack the tensor from the tuple for downstream stacking
            "labels": y,
            "predictions": y_hat,
            "conf_matrix_aux": conf_matrix_aux,
            "val_acc_aux": val_acc_aux,
        }
        # PL 2.0 FIX: Store step output
        self.validation_step_outputs.append(output)
        return output

    def training_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        inputs, targets = batch
        model_out = self.forward(inputs)
        loss_val = self.loss(model_out, targets)

        if self.hparams.aux_task != "None":
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

        y = targets["labels"]
        y_hat = model_out["logits"]

        if self.hparams.aux_task != "None":
            val_acc_aux, conf_matrix_aux = self._compute_aux_metrics(
                targets["labels_aux"],
                model_out["logits_aux"],
                device=loss_val[0].device,  # FIX: was loss_val.device — loss() returns a tuple
                dataset_columns=self._test_dataset.columns,  # FIX: test uses _test_dataset not _dev_dataset
            )
        else:
            val_acc_aux = torch.zeros(1, device=loss_val[0].device)
            conf_matrix_aux = torch.zeros(1, 1, device=loss_val[0].device)

        output = {
            "val_loss": loss_val[0],  # FIX: unpack tensor from tuple
            "labels": y,
            "predictions": y_hat,
            "conf_matrix_aux": conf_matrix_aux,
            "val_acc_aux": val_acc_aux,
        }
        # PL 2.0 FIX: Store step output
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

        # --- Accumulate losses and aux accuracy ---
        val_loss_mean = torch.stack([o["val_loss"] for o in outputs]).mean()
        val_acc_aux_mean = torch.stack([o["val_acc_aux"] for o in outputs]).mean()

        # --- Concatenate predictions and labels (stay on device) ---
        val_y = torch.cat([o["labels"] for o in outputs])  # (N,)
        val_y_hat = torch.cat([o["predictions"] for o in outputs])  # (N,)

        # --- Sum confusion matrices ---
        conf_matrix_aux = torch.stack([o["conf_matrix_aux"] for o in outputs]).sum(dim=0)

        # --- Pearson correlation (torch-native, no numpy, no cpu move) ---
        def _pearson(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x = x.float().flatten()
            y = y.float().flatten()
            x_mean = x - x.mean()
            y_mean = y - y.mean()
            return (x_mean * y_mean).sum() / (
                    x_mean.norm() * y_mean.norm() + 1e-8
            )

        pearsonr = _pearson(val_y, val_y_hat)

        # --- Log metrics (Lightning handles DDP syncing automatically) ---
        self.log("val_loss", val_loss_mean, prog_bar=True, sync_dist=True)
        self.log("val_pearson", pearsonr, prog_bar=True, sync_dist=True)
        self.log("val_acc_aux", val_acc_aux_mean, prog_bar=True, sync_dist=True)

        # --- Confusion matrix figures (only on rank 0 to avoid duplicates) ---
        if self.trainer.is_global_zero and self.hparams.aux_task != "None":
            labels = (
                self._dev_dataset.columns
                if self.hparams.aux_task == "emotions"
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
            plt.close(fig)  # close only this figure, not all

        # PL 2.0 FIX: Clear memory
        self.validation_step_outputs.clear()

    # PL 2.0 FIX: Renamed test_epoch_end to on_test_epoch_end and removed outputs arg
    def on_test_epoch_end(self) -> None:
        """Aggregates test step outputs and logs metrics."""
        outputs = self.test_step_outputs

        # torch.stack creates a 2D tensor from all the scalar losses,
        # then .mean() reduces it to a single scalar — no manual loop needed
        val_loss_mean = torch.stack([o["val_loss"] for o in outputs]).mean()

        # torch.cat glues all the per-batch tensors along dim=0,
        # giving us one big (N,) tensor for the whole test set — stays on device
        val_y = torch.cat([o["labels"] for o in outputs])
        val_y_hat = torch.cat([o["predictions"] for o in outputs])

        # Same idea: stack adds a new dim-0, sum(dim=0) collapses it,
        # leaving one matrix of shape (num_classes, num_classes)
        conf_matrix_aux = torch.stack([o["conf_matrix_aux"] for o in outputs]).sum(dim=0)

        # Pearson on-device (no .cpu() / numpy needed)
        # We subtract the mean to center, then use the dot-product formula:
        #   r = sum((x - x̄)(y - ȳ)) / (||x - x̄|| * ||y - ȳ||)
        # 1e-8 guards against division by zero on constant predictions
        def _pearson(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x = x.float().flatten()
            y = y.float().flatten()
            x_c = x - x.mean()
            y_c = y - y.mean()
            return (x_c * y_c).sum() / (x_c.norm() * y_c.norm() + 1e-8)

        pearsonr = _pearson(val_y, val_y_hat)

        # self.log() is the modern Lightning API for metrics:
        # - prog_bar=True  → shows in the tqdm bar
        # - sync_dist=True → averages the value across GPUs in DDP automatically
        #                    (replaces the old manual use_dp / use_ddp2 reduction)
        self.log("test_loss", val_loss_mean, prog_bar=True, sync_dist=True)
        self.log("test_pearson", pearsonr, prog_bar=True, sync_dist=True)

        # Save predictions to CSV — .cpu() here is unavoidable since
        # csv.writer and numpy cannot read GPU tensors
        val_labels = val_y.float().cpu().numpy().flatten()
        val_preds = val_y_hat.float().cpu().numpy().flatten()
        pred_path = Path(self.hparams.checkpoint_path) / "predictions.csv"
        with pred_path.open("w", newline="") as f:
            csv.writer(f).writerows(zip(val_preds, val_labels))

        # is_global_zero ensures only one process logs/plots in multi-GPU runs
        # Without this, every GPU would write the same figure to the logger
        if self.trainer.is_global_zero and self.hparams.aux_task != "None":
            labels = (
                self._test_dataset.columns
                if self.hparams.aux_task == "emotions"
                else self.hparams.le_aux.classes_
            )
            # .float() keeps the heatmap on CPU-friendly dtype for seaborn;
            # seaborn can't read CUDA tensors so .cpu() is unavoidable here too
            fig, ax = plt.subplots(figsize=(10, 7))
            sn.heatmap(conf_matrix_aux.float().cpu(), annot=True, ax=ax)
            ax.set_xlabel("Predicted labels")
            ax.set_ylabel("True labels")
            ax.set_title("Confusion Matrix")
            ax.xaxis.set_ticklabels(labels)
            ax.yaxis.set_ticklabels(labels)
            self.logger.experiment.add_figure("confusion matrix Aux", fig)
            plt.close(fig)  # close only this figure, not every open matplotlib window

        # PL 2.0 FIX: Clear memory
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        """Sets different learning rates for different parameter groups."""

        if self.hparams.aux_task != "None" and self.hparams.gradnorm:
            param_groups = [
                # FIX: Changed learning_rate to encoder_learning_rate
                {"params": self.model.parameters(), "lr": self.hparams.encoder_learning_rate},
                {"params": self.weights, "lr": 1e-2},
            ]
        elif self.hparams.aux_task == "emotions":
            aux_keywords = {"classification_head_aux", "dense_emotions", "layer_emotion"}
            params, aux_params = [], []
            for name, param in self.model.named_parameters():
                if any(kw in name for kw in aux_keywords):
                    aux_params.append(param)
                else:
                    params.append(param)
            param_groups = [
                # FIX: Changed learning_rate to encoder_learning_rate
                {"params": params, "lr": self.hparams.encoder_learning_rate},
                {"params": aux_params, "lr": self.hparams.encoder_learning_rate * 10},
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
            default="data/UsVsThem_train_public.csv",
            type=str,
            help="Path to the file containing the train data.",
        )
        parser.add_argument(
            "--dev_csv",
            default="data/UsVsThem_valid_public.csv",
            type=str,
            help="Path to the file containing the train data.",
        )
        parser.add_argument(
            "--test_csv",
            default="data/UsVsThem_test_public.csv",
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