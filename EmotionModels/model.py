import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AutoModelForSequenceClassification
from sklearn.metrics import jaccard_score
import logging
import pandas as pd

from dataloader import sentiment_analysis_dataset, MyCollator

log = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """
    Binary Focal Loss for multi-label classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma      : focusing parameter — higher = more focus on hard examples.
                     gamma=0 reduces to standard BCE. Typical values: 1–3.
        pos_weight : per-label weight tensor (same as BCEWithLogitsLoss.pos_weight).
                     Addresses class imbalance on top of focal modulation.
        reduction  : 'mean' | 'sum' | 'none'
    """

    def __init__(
            self,
            gamma:float = 2.0,
            pos_weight: torch.tensor | None = None,
            reduction: str = "mean"
    ):
        
        super.__init__()

        self.gamma = gamma
        self.reduction = reduction


def compute_weights_from_csv(csv_paths,clamp_max = 20.0):

    dfs = []

    for path in csv_paths:
        if os.path.exists(path):
            dfs.append(pd.read_csf(path))

    if not dfs:
        log.warning("No csv found")
        raise Exception("No pandas found")
    

    df_all = pd.concat(dfs,ignore_index=True)
    exclude_cols = {'body', 'usVSthem_scale', 'is_Disc_Crit', 'group', 'bias', 'allsides_name', 'Unnamed: 0'}
    label_cols = [c for c in df_all.columns if c not in exclude_cols]

    total = len(df_all)
    weights = []

    for col in label_cols:
        # Convert to numeric (handles boolean True/False or 1/0 appropriately)
        c = pd.to_numeric(df_all[col], errors='coerce').sum()
        if pd.isna(c) or c == 0:
            weights.append(10.0)  # Conservative default for extremely rare/missing
        else:
            weights.append((total - c) / c)

    w = torch.tensor(weights, dtype=torch.float)
    return torch.clamp(w, max=clamp_max), total, label_cols

class EmotionModel(pl.LightningModule):
    def __init__(self, hparams) -> None:
        super(EmotionModel, self).__init__()

        # Clean hparams to prevent TensorBoard/Logger crashes with non-scalar types
        if hasattr(hparams, '__dict__'):
            hparams_dict = vars(hparams)
        else:
            hparams_dict = dict(hparams)

        clean_hparams = {
            k: v for k, v in hparams_dict.items()
            if isinstance(v, (int, float, str, bool, type(None)))
            k: v for k, v in hparams_dict.items()
            if isinstance(v, (int, float, str, bool, type(None)))
        }
        self.save_hyperparameters(clean_hparams)

        # State management for Lightning 2.0+
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # Collator and model
        self.prepare_sample = MyCollator(self.hparams.encoder_model, self.hparams.max_length)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.hparams.encoder_model)

        # Label Expansion Mapping (13 Custom Labels -> 28 GoEmotions Nodes)
        # Maps indices: Anger(0), Contempt(1), Disgust(2), Fear(3), Gratitude(4), Guilt(5),
        # Happiness(6), Hope(7), Pride(8), Relief(9), Sadness(10), Sympathy(11), Neutral(12)
        self.register_buffer('mapping', torch.tensor([
            6, 6, 0, 0, 12, 11, 12, 12, 7, 10, 1, 2, 5, 6, 3, 4,
            10, 6, 6, 3, 7, 8, 12, 9, 5, 10, 12, 12
        ], dtype=torch.long))
        # FIX: register_buffer ensures `self.mapping` is automatically moved to the
        # correct device alongside the model — no manual .to(device) calls needed.

        self.loss_fn = nn.BCEWithLogitsLoss()

    # FIX: Removed setup() and on_train_start() overrides that manually called
    # module.train() on every submodule. Lightning already manages train/eval mode
    # correctly; overriding it can break BatchNorm and Dropout behaviour.

    def _safe_squeeze(self, inputs):
        """
        Safely removes an extra batch dimension of size 1 that some DataLoader
        configurations add, without collapsing valid single-token sequences.
        """
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        if input_ids.dim() == 3 and input_ids.size(1) == 1:
            input_ids = input_ids.squeeze(1)
        if attention_mask.dim() == 3 and attention_mask.size(1) == 1:
            attention_mask = attention_mask.squeeze(1)

        return input_ids, attention_mask

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    def calculate_loss(self, logits_28, targets_13):
        # self.mapping is already on the right device via register_buffer
        targets_28 = targets_13[:, self.mapping]
        loss = self.loss_fn(logits_28, targets_28)
        return loss, targets_28

    def training_step(self, batch, batch_nb):
        inputs, targets = batch
        input_ids, attention_mask = self._safe_squeeze(inputs)

        logits_28 = self.forward(input_ids, attention_mask)
        loss, _ = self.calculate_loss(logits_28, targets['labels_aux'])

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.training_step_outputs.append({"loss": loss})
        return loss
        self.training_step_outputs.append({"loss": loss})
        return loss

    def validation_step(self, batch, batch_nb):
    def validation_step(self, batch, batch_nb):
        inputs, targets = batch
        input_ids, attention_mask = self._safe_squeeze(inputs)
        input_ids, attention_mask = self._safe_squeeze(inputs)

        logits_28 = self.forward(input_ids, attention_mask)
        loss, targets_28 = self.calculate_loss(logits_28, targets['labels_aux'])

        preds_28 = (logits_28 > 0).float()

        output = {
            "val_loss": loss,
            "preds": preds_28,
            "targets": targets_28,
        }
        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()

        all_preds = torch.cat([x["preds"] for x in self.validation_step_outputs], dim=0)
        all_targets = torch.cat([x["targets"] for x in self.validation_step_outputs], dim=0)

        all_preds_np = all_preds.cpu().numpy()
        all_targets_np = all_targets.cpu().numpy()

        global_jaccard = jaccard_score(
            all_targets_np,
            all_preds_np,
            average="macro",
            zero_division=0,
        )

        self.log("val_loss", avg_loss, prog_bar=True, sync_dist=True)
        self.log("val_jaccard", global_jaccard, prog_bar=True, sync_dist=True)

        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_nb):
        # FIX: Added test_step so trainer.test() doesn't crash or silently reuse
        # val data. Mirrors validation_step logic.
        inputs, targets = batch
        input_ids, attention_mask = self._safe_squeeze(inputs)

        logits_28 = self.forward(input_ids, attention_mask)
        loss, targets_28 = self.calculate_loss(logits_28, targets['labels_aux'])

        preds_28 = (logits_28 > 0).float()

        output = {
            "test_loss": loss,
            "preds": preds_28,
            "targets": targets_28,
        }
        self.test_step_outputs.append(output)
        return output

    def on_test_epoch_end(self):
        avg_loss = torch.stack([x["test_loss"] for x in self.test_step_outputs]).mean()

        all_preds = torch.cat([x["preds"] for x in self.test_step_outputs], dim=0)
        all_targets = torch.cat([x["targets"] for x in self.test_step_outputs], dim=0)

        all_preds_np = all_preds.cpu().numpy()
        all_targets_np = all_targets.cpu().numpy()

        global_jaccard = jaccard_score(
            all_targets_np,
            all_preds_np,
            average="macro",
            zero_division=0,
        )

        self.log("test_loss", avg_loss, prog_bar=True)
        self.log("test_jaccard", global_jaccard, prog_bar=True)

        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.encoder_learning_rate)

        # FIX: Compute train steps directly from the dataset length instead of
        # instantiating a second DataLoader just to call len() on it.
        dataset = sentiment_analysis_dataset(self.hparams, train=True, val=False, test=False)
        steps_per_epoch = len(dataset) // self.hparams.batch_size
        train_steps = steps_per_epoch * self.hparams.max_epochs
        warmup_steps = int(self.hparams.warmup_proportion * train_steps)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=train_steps,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    # ---------- DataLoaders ----------

    def train_dataloader(self):
        dataset = sentiment_analysis_dataset(self.hparams, train=True, val=False, test=False)
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
            # FIX: pin_memory speeds up CPU->GPU transfers when using a GPU
            pin_memory=self.hparams.loader_workers > 0,
            # FIX: persistent_workers avoids re-spawning worker processes each epoch
            persistent_workers=self.hparams.loader_workers > 0,
        )

    def val_dataloader(self):
        dataset = sentiment_analysis_dataset(self.hparams, train=False, val=True, test=False)
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
            pin_memory=self.hparams.loader_workers > 0,
            persistent_workers=self.hparams.loader_workers > 0,
        )

    def test_dataloader(self):
        # FIX: Added missing test_dataloader — required for trainer.test() to work
        dataset = sentiment_analysis_dataset(self.hparams, train=False, val=False, test=True)
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
            pin_memory=self.hparams.loader_workers > 0,
            persistent_workers=self.hparams.loader_workers > 0,
        )

    @classmethod
    def add_model_specific_args(cls, parser):
        parser.add_argument("--encoder_model", default="SamLowe/roberta-base-go_emotions", type=str)
        parser.add_argument("--encoder_learning_rate", default=2e-5, type=float)
        parser.add_argument("--warmup_proportion", default=0.1, type=float)
        parser.add_argument("--max_length", default=128, type=int)
        # FIX: Changed default from 0 to 4. Use 0 only on Windows where
        # multiprocessing with CUDA can cause issues.
        parser.add_argument("--loader_workers", default=4, type=int)
        parser.add_argument("--train_csv", default="Resources/UsVsThem_train_public.csv", type=str)
        parser.add_argument("--dev_csv", default="Resources/UsVsThem_valid_public.csv", type=str)
        parser.add_argument("--test_csv", default="Resources/UsVsThem_test_public.csv", type=str)
        return parser