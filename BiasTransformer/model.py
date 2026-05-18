import os

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from dataloader import BiasCollator, BiasDataset


class BiasTransformer(pl.LightningModule):
    def __init__(self, hparams=None, **kwargs):
        super().__init__()
        if hparams is None:
            import argparse
            hparams = argparse.Namespace(**kwargs)

        hp = vars(hparams) if hasattr(hparams, "__dict__") else dict(hparams)
        self.save_hyperparameters({k: v for k, v in hp.items() if isinstance(v, (int, float, str, bool, type(None)))})

        config = AutoConfig.from_pretrained(
            self.hparams.encoder_model,
            num_labels=2,
            id2label={0: "Non-biased", 1: "Biased"},
            label2id={"Non-biased": 0, "Biased": 1},
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.hparams.encoder_model,
            config=config,
            ignore_mismatched_sizes=True,
        )
        self.collator = BiasCollator(self.hparams.encoder_model, self.hparams.max_length)
        self.validation_outputs = []
        self.test_outputs = []

    def forward(self, **inputs):
        return self.model(**inputs).logits

    def _step(self, batch):
        inputs, targets = batch
        logits = self.forward(**inputs)
        loss_per_row = F.cross_entropy(logits, targets["labels"], reduction="none")
        weights = targets["sample_weight"].to(loss_per_row.device)
        loss = (loss_per_row * weights).sum() / weights.sum().clamp(min=1e-6)
        preds = logits.argmax(dim=-1)
        return loss, preds.detach(), targets["labels"].detach()

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._step(batch)
        self.validation_outputs.append({"loss": loss.detach(), "preds": preds.cpu(), "labels": labels.cpu()})
        return loss

    def on_validation_epoch_end(self):
        self._log_epoch_metrics(self.validation_outputs, "val")
        self.validation_outputs.clear()

    def test_step(self, batch, batch_idx):
        loss, preds, labels = self._step(batch)
        self.test_outputs.append({"loss": loss.detach(), "preds": preds.cpu(), "labels": labels.cpu()})
        return loss

    def on_test_epoch_end(self):
        self._log_epoch_metrics(self.test_outputs, "test", print_report=True)
        self.test_outputs.clear()

    def _log_epoch_metrics(self, outputs, prefix: str, print_report: bool = False):
        if not outputs:
            return
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        preds = torch.cat([x["preds"] for x in outputs]).numpy()
        labels = torch.cat([x["labels"] for x in outputs]).numpy()

        acc = accuracy_score(labels, preds)
        f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
        f1_biased = f1_score(labels, preds, average="binary", pos_label=1, zero_division=0)

        self.log(f"{prefix}_loss", loss, prog_bar=True)
        self.log(f"{prefix}_accuracy", acc, prog_bar=True)
        self.log(f"{prefix}_f1_macro", f1_macro, prog_bar=True)
        self.log(f"{prefix}_f1_biased", f1_biased, prog_bar=False)

        if print_report:
            print(classification_report(labels, preds, target_names=["Non-biased", "Biased"], zero_division=0))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        steps_per_epoch = max(1, len(self.train_dataloader()))
        total_steps = steps_per_epoch * self.hparams.max_epochs
        warmup_steps = int(total_steps * self.hparams.warmup_proportion)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def train_dataloader(self):
        return self._loader(self.hparams.train_csv, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.hparams.dev_csv, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.hparams.test_csv, shuffle=False)

    def _loader(self, csv_path: str, shuffle: bool):
        dataset = BiasDataset(csv_path)
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            collate_fn=self.collator,
            num_workers=self.hparams.loader_workers,
            pin_memory=self.hparams.loader_workers > 0,
            persistent_workers=self.hparams.loader_workers > 0,
        )

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--encoder_model", default="roberta-base")
        parser.add_argument("--max_length", type=int, default=256)
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--warmup_proportion", type=float, default=0.10)
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--loader_workers", type=int, default=0)
        parser.add_argument("--train_csv", default="data/bias_transformer/finetune_train.csv")
        parser.add_argument("--dev_csv", default="data/bias_transformer/finetune_valid.csv")
        parser.add_argument("--test_csv", default="data/bias_transformer/finetune_test.csv")
        return parser
