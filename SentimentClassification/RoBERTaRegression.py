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
        
        # Initialize model
        self.model = RoBERTaMTL(
            model_name=self.hparams.encoder_model,
            num_emotions=13,  # Standard for this project
            num_social_groups=10,
            extra_dropout=getattr(self.hparams, 'extra_dropout', 0.1),
            loss_weights={
                "bias": 1.0, 
                "emotion": 1.0, 
                "social": 0.5 if getattr(self.hparams, 'ablate_social_group', False) else 1.0
            }
        )
        
        # Set initial ablation state
        if getattr(self.hparams, 'ablate_social_group', False):
            self.model.set_ablation_mode(True)
        
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
        
        # Pass labels directly to the model for internal loss calculation
        outputs = self.forward(
            inputs, 
            labels_bias=targets.get("labels"),
            labels_emotion=targets.get("labels_aux"),
            labels_social=targets.get("labels_social") if "labels_social" in targets else None
        )
        
        total_loss = outputs["loss"]
        
        self.log("train_loss", total_loss, prog_bar=True, sync_dist=True)
        self.training_step_outputs.append({"loss": total_loss})
        return total_loss

    def validation_step(self, batch: tuple, batch_nb: int) -> dict:
        inputs, targets = batch
        
        outputs = self.forward(
            inputs, 
            labels_bias=targets.get("labels"),
            labels_emotion=targets.get("labels_aux"),
        )
        
        # Calculate accuracy for emotions (aux task)
        # Using Jaccard score for multi-label
        y_aux = targets["labels_aux"].cpu().numpy()
        y_aux_hat = (torch.sigmoid(outputs["emotions"]) > 0.5).cpu().numpy()
        val_acc_aux = jaccard_score(y_aux, y_aux_hat, average="macro")
        
        output = {
            "val_loss": outputs["loss"],
            "labels": targets["labels"],
            "predictions": outputs["bias_score"],
            "val_acc_aux": torch.tensor(val_acc_aux)
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
        predictions = self.forward(inputs)
        losses = self.loss(predictions, targets)
        
        output = {
            "test_loss": losses["bias"] + losses["emotion"],
            "labels": targets["labels"],
            "predictions": predictions["bias_score"]
        }
        self.test_step_outputs.append(output)
        return output

    def on_test_epoch_end(self) -> None:
        outputs = self.test_step_outputs
        
        all_labels = torch.cat([o["labels"] for o in outputs]).cpu().numpy().flatten()
        all_preds = torch.cat([o["predictions"] for o in outputs]).cpu().numpy().flatten()
        
        # Save predictions
        pred_path = Path(self.hparams.checkpoint_path) / "predictions.csv"
        with pred_path.open("w", newline="") as f:
            csv.writer(f).writerows(zip(all_preds, all_labels))
            
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
