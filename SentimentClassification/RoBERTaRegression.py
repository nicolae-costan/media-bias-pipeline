import csv
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import jaccard_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sn
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from transformers import get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder

from dataloader import sentiment_analysis_dataset, MyCollator
from RoBERTaMTL import RoBERTaMTL

class RoBERTaRegressor(pl.LightningModule):
    """
    LightningModule wrapper for RoBERTaMTL.
    Optimized for multi-task sentiment and bias analysis.
    """
    
    def __init__(self, hparams) -> None:
        super(RoBERTaRegressor, self).__init__()
        
        # Clean hparams for TensorBoard
        if hasattr(hparams, '__dict__'):
            hparams_dict = vars(hparams)
        else:
            hparams_dict = dict(hparams)
            
        clean_hparams = {
            k: v for k, v in hparams_dict.items()
            if isinstance(v, (int, float, str, bool, type(None)))
        }
        self.save_hyperparameters(clean_hparams)
        
        # Track outputs for epoch-end aggregation
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
        # Collator
        self.prepare_sample = MyCollator(
            self.hparams.encoder_model, 
            self.hparams.max_length
        )
        
        # FIX: DYNAMIC MODEL BUILDING
        # 1. Fit label encoders to data
        # 2. Initialize RoBERTaMTL with correct head sizes
        self.__build_model()
        
        # Set initial ablation state
        if getattr(self.hparams, 'ablate_social_group', False):
            self.model.set_ablation_mode(True)
        
    def __build_model(self):
        """
        Dynamically adjusts head sizes based on the dataset before initializing RoBERTaMTL.
        """
        try:
            train_df = pd.read_csv(self.hparams.train_csv)
            test_df = pd.read_csv(self.hparams.test_csv)
            dev_df = pd.read_csv(self.hparams.dev_csv)
            comments = pd.concat([train_df, test_df, dev_df])
        except Exception as e:
            print(f"Could not load CSV for sizing check: {e}")
            # Fallback to defaults
            comments = None

        self.hparams.le = LabelEncoder()
        self.hparams.le_aux = LabelEncoder()

        aux_task_str = str(self.hparams.aux_task)
        num_emotions = 13 # Standard
        num_social = 10 # Default fallback

        if aux_task_str == 'emotions':
            # Emotions are multi-label (one-hot), no le_aux needed
            pass 
        elif aux_task_str not in ('None', 'bias'): # e.g. 'group'
            if comments is not None:
                self.hparams.le_aux.fit(comments[self.hparams.aux_task].values)
                num_social = len(self.hparams.le_aux.classes_)
                print(f"--- Detected {num_social} classes for task: {aux_task_str} ---")
        
        self.model = RoBERTaMTL(
            model_name=self.hparams.encoder_model,
            num_emotions=num_emotions,
            num_social_groups=num_social,
            extra_dropout=getattr(self.hparams, 'extra_dropout', 0.1),
            loss_weights={
                "bias": 1.0, 
                "emotion": 1.0, 
                "social": 0.5 if getattr(self.hparams, 'ablate_social_group', False) else 1.0
            }
        )

    def forward(
        self, 
        batch_inputs: dict, 
        labels_bias: torch.Tensor = None, 
        labels_emotion: torch.Tensor = None, 
        labels_social: torch.Tensor = None
    ) -> dict:
        """Process inputs and labels via internal RoBERTaMTL logic."""
        return self.model(
            input_ids=batch_inputs["input_ids"],
            attention_mask=batch_inputs["attention_mask"],
            labels_bias=labels_bias,
            labels_emotion=labels_emotion,
            labels_social=labels_social
        )

    def training_step(self, batch: tuple, batch_nb: int) -> torch.Tensor:
        inputs, targets = batch
        
        # SMART ROUTING: Pass labels to the correct head
        aux_task_str = str(self.hparams.aux_task)
        labels_emotion = targets.get("labels_aux") if aux_task_str == 'emotions' else None
        labels_social = targets.get("labels_aux") if aux_task_str not in ('emotions', 'None', 'bias') else None

        outputs = self.forward(
            inputs, 
            labels_bias=targets.get("labels"),
            labels_emotion=labels_emotion,
            labels_social=labels_social
        )
        
        total_loss = outputs["loss"]
        
        self.log("train_loss", total_loss, prog_bar=True, sync_dist=True)
        self.training_step_outputs.append({"loss": total_loss})
        return total_loss

    def _compute_aux_metrics(self, y_aux: torch.Tensor, outputs: dict) -> torch.Tensor:
        """Calculates accuracy for the active auxiliary task."""
        aux_task_str = str(self.hparams.aux_task)
        if aux_task_str == 'emotions':
            # Multi-label Jaccard score
            y_aux_np = y_aux.cpu().numpy()
            y_aux_hat_np = (torch.sigmoid(outputs["emotions"]) > 0.5).cpu().numpy()
            acc = jaccard_score(y_aux_np, y_aux_hat_np, average="macro")
            return torch.tensor(acc)
        elif aux_task_str not in ('None', 'bias'):
            # Single-label classification accuracy
            y_aux_hat = torch.argmax(outputs["social_group"], dim=1)
            acc = (y_aux.long() == y_aux_hat).float().mean()
            return acc
        else:
            return torch.tensor(0.0)

    def validation_step(self, batch: tuple, batch_nb: int) -> dict:
        inputs, targets = batch
        
        # SMART ROUTING
        aux_task_str = str(self.hparams.aux_task)
        labels_emotion = targets.get("labels_aux") if aux_task_str == 'emotions' else None
        labels_social = targets.get("labels_aux") if aux_task_str not in ('emotions', 'None', 'bias') else None

        outputs = self.forward(
            inputs, 
            labels_bias=targets.get("labels"),
            labels_emotion=labels_emotion,
            labels_social=labels_social,
        )
        
        val_acc_aux = self._compute_aux_metrics(targets.get("labels_aux"), outputs)
        
        output = {
            "val_loss": outputs["loss"],
            "labels": targets["labels"],
            "predictions": outputs["bias_score"],
            "val_acc_aux": val_acc_aux
        }
        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self) -> None:
        outputs = self.validation_step_outputs
        
        avg_loss = torch.stack([o["val_loss"] for o in outputs]).mean()
        avg_acc_aux = torch.stack([o["val_acc_aux"] for o in outputs]).mean()
        
        # Pearson Correlation for Bias
        all_labels = torch.cat([o["labels"] for o in outputs]).flatten()
        all_preds = torch.cat([o["predictions"] for o in outputs]).flatten()
        
        pearsonr = self._pearson(all_labels, all_preds)
        
        self.log("val_loss", avg_loss, prog_bar=True, sync_dist=True)
        self.log("val_pearson", pearsonr, prog_bar=True, sync_dist=True)
        self.log("val_acc_aux", avg_acc_aux, prog_bar=True, sync_dist=True)
        
        self.validation_step_outputs.clear()

    def test_step(self, batch: tuple, batch_nb: int) -> dict:
        # Similar logic to validation
        inputs, targets = batch
        
        # SMART ROUTING
        aux_task_str = str(self.hparams.aux_task)
        labels_emotion = targets.get("labels_aux") if aux_task_str == 'emotions' else None
        labels_social = targets.get("labels_aux") if aux_task_str not in ('emotions', 'None', 'bias') else None

        outputs = self.forward(
            inputs,
            labels_bias=targets.get("labels"),
            labels_emotion=labels_emotion,
            labels_social=labels_social
        )
        
        test_acc_aux = self._compute_aux_metrics(targets.get("labels_aux"), outputs)

        output = {
            "test_loss": outputs["loss"],
            "labels": targets["labels"],
            "predictions": outputs["bias_score"],
            "test_acc_aux": test_acc_aux
        }
        self.test_step_outputs.append(output)
        return output

    def on_test_epoch_end(self) -> None:
        outputs = self.test_step_outputs
        
        # 1. Aggregated Metrics
        avg_loss = torch.stack([o["test_loss"] for o in outputs]).mean()
        avg_acc_aux = torch.stack([o["test_acc_aux"] for o in outputs]).mean()
        
        all_labels = torch.cat([o["labels"] for o in outputs]).flatten()
        all_preds = torch.cat([o["predictions"] for o in outputs]).flatten()
        
        # Pearson Correlation for Bias
        pearsonr = self._pearson(all_labels, all_preds)
        
        # Log to show in the final Trainer table
        self.log("test_loss", avg_loss, prog_bar=True)
        self.log("test_pearson", pearsonr, prog_bar=True)
        self.log("test_acc_aux", avg_acc_aux, prog_bar=True)
        
        # 2. Save predictions to CSV
        pred_path = Path(self.hparams.checkpoint_path) / "predictions.csv"
        with pred_path.open("w", newline="") as f:
            csv.writer(f).writerows(zip(all_preds.cpu().numpy(), all_labels.cpu().numpy()))
            
        print(f"\n--- Test Results: Loss={avg_loss:.4f}, Pearson={pearsonr:.4f}, Aux Acc={avg_acc_aux:.4f} ---")
        self.test_step_outputs.clear()

    def _pearson(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.float().flatten()
        y = y.float().flatten()
        x_m = x - x.mean()
        y_m = y - y.mean()
        return (x_m * y_m).sum() / (x_m.norm() * y_m.norm() + 1e-8)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.encoder_learning_rate)
        
        # Simple warmup scheduler
        train_steps = len(self.train_dataloader()) * self.hparams.max_epochs
        warmup_steps = int(getattr(self.hparams, 'warmup_proportion', 0.1) * train_steps)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=train_steps,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def train_dataloader(self) -> DataLoader:
        self._train_dataset = sentiment_analysis_dataset(self.hparams, val=False, test=False)
        return DataLoader(
            self._train_dataset,
            sampler=RandomSampler(self._train_dataset),
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=getattr(self.hparams, 'loader_workers', 0)
        )

    def val_dataloader(self) -> DataLoader:
        self._dev_dataset = sentiment_analysis_dataset(self.hparams, train=False, test=False)
        return DataLoader(
            self._dev_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=getattr(self.hparams, 'loader_workers', 0)
        )

    def test_dataloader(self) -> DataLoader:
        self._test_dataset = sentiment_analysis_dataset(self.hparams, train=False, val=False)
        return DataLoader(
            self._test_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=getattr(self.hparams, 'loader_workers', 0)
        )

    @classmethod
    def add_model_specific_args(cls, parser):
        # We reuse the same args structure to ensure compatibility
        return parser
