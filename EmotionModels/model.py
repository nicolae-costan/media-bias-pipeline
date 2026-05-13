import math
import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AutoModel, AutoConfig
from sklearn.metrics import jaccard_score
import logging
import pandas as pd
import json


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
            pos_weight: torch.Tensor | None = None,
            reduction: str = "mean"
    ):
        
        super().__init__()

        self.gamma = gamma
        self.reduction = reduction
        if pos_weight is not None:
            self.register_buffer('pos_weight', pos_weight)
        else:
            self.pos_weight = None

    def forward(self,logits:torch.Tensor,targets:torch.Tensor):


        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,targets,pos_weight=self.pos_weight,reduction='none'
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)

        focal_weight = (1.0-p_t)**self.gamma
        loss = focal_weight * bce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) for multi-label classification.
    Designed to handle severe class imbalance by decoupling positive and negative focusing.
    
    L = -y(1-p)^gamma_pos * log(p) - (1-y)p_m^gamma_neg * log(1-p_m)
    where p_m = shifted/clipped probability for negative samples.
    """
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, pos_weight=None):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        if pos_weight is not None:
            self.register_buffer('pos_weight', pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits, targets):
        # Calculate probabilities
        probs = torch.sigmoid(logits)
        
        # Positive samples
        loss_pos = targets * torch.log(probs.clamp(min=self.eps))
        if self.gamma_pos > 0:
            loss_pos *= (1 - probs) ** self.gamma_pos
            
        # Negative samples
        # Asymmetric Clipping/Probability shifting (pm)
        probs_neg = 1 - probs
        if self.clip is not None and self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1)
            
        loss_neg = (1 - targets) * torch.log(probs_neg.clamp(min=self.eps))
        if self.gamma_neg > 0:
            # Note: in ASL, we use the shifted probability (probs_neg) for the focusing term
            loss_neg *= (1 - probs_neg) ** self.gamma_neg
            
        loss = -(loss_pos + loss_neg)
        
        if self.pos_weight is not None:
            loss *= self.pos_weight
            
        return loss.mean()


def compute_weights_from_csv(csv_paths,clamp_max = 20.0):

    dfs = []

    for path in csv_paths:
        if not os.path.exists(path):
            # Universally resolve against the project root's data directory
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            alt_path = os.path.join(project_root, "data", os.path.basename(path))
            if os.path.exists(alt_path):
                path = alt_path
        
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))

    if not dfs:
        log.warning("No csv found")
        raise Exception("No pandas found")
    

    df_all = pd.concat(dfs,ignore_index=True)
    label_cols = [
        'Anger', 'Contempt', 'Disgust', 'Fear', 'Gratitude',
        'Guilt', 'Happiness', 'Hope', 'Pride', 'Relief',
        'Sadness', 'Sympathy', 'Emotions_Neutral'
    ]
    total = len(df_all)
    weights = []

    for col in label_cols:
        c = pd.to_numeric(df_all[col], errors='coerce').sum()
        neg = total - c
        if pd.isna(c) or c == 0:
            # Label never appears — conservative default so there is still a
            # gradient signal if it shows up at inference time
            weights.append(10.0)
        elif neg == 0:
            weights.append(clamp_max)
        else:
            # log(1 + neg/pos) instead of raw neg/pos:
            #   - still up-weights rare labels (higher ratio → higher weight)
            #   - logarithm compresses the extreme tail so a label that appears
            #     in 1% of rows gets weight ≈ 4.6 instead of ≈ 99, preventing
            #     gradient spikes without needing an aggressive hard clamp
            weights.append(math.log1p(float(neg) / float(c)))

    w = torch.tensor(weights, dtype=torch.float)
    return torch.clamp(w, max=clamp_max), total, label_cols

class EmotionModel(pl.LightningModule):


    def __init__(self, hparams=None, **kwargs) -> None:
        super(EmotionModel, self).__init__()

        # If loaded from checkpoint, Lightning passes hyperparameters as kwargs
        if hparams is None:
            import argparse
            hparams = argparse.Namespace(**kwargs)

        # Clean hparams to prevent TensorBoard/Logger crashes with non-scalar types
        if hasattr(hparams, '__dict__'):
            hparams_dict = vars(hparams)
        else:
            hparams_dict = dict(hparams)

        clean_hparams = {
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

        # MInT pooling: load the bare encoder (no classification head) so we can
        # apply mean pooling ourselves over all non-padding token embeddings.
        # This replaces the previous CLS-only pooling from AutoModelForSequenceClassification.
        config = AutoConfig.from_pretrained(self.hparams.encoder_model)
        self.model = AutoModel.from_pretrained(
            self.hparams.encoder_model,
            config=config,
        )
        hidden_size = config.hidden_size
        # Fresh 13-class linear head (replaces the discarded pretrained head)
        self.classifier = nn.Linear(hidden_size, 13)

        pos_weight_13, _, _ = compute_weights_from_csv(
            [self.hparams.train_csv, self.hparams.dev_csv],
            clamp_max=getattr(self.hparams, 'focal_weight_clamp', 20.0),
        )

        if getattr(self.hparams, 'use_asl', True):
            self.loss_fn = AsymmetricLoss(
                gamma_neg=getattr(self.hparams, 'asl_gamma_neg', 4.0),
                gamma_pos=getattr(self.hparams, 'asl_gamma_pos', 1.0),
                clip=getattr(self.hparams, 'asl_clip', 0.05),
                pos_weight=pos_weight_13
            )
        else:
            self.loss_fn = FocalLoss(
                gamma=getattr(self.hparams, 'focal_gamma', 2.0),
                pos_weight=pos_weight_13,
                reduction='mean',
            )

        # Default thresholds (0.5 for all 13 nodes)
        self.register_buffer('thresholds', torch.ones(13, dtype=torch.float) * 0.5)

    def load_thresholds(self, thresholds_path):
        if os.path.exists(thresholds_path):
            with open(thresholds_path, 'r') as f:
                t_list = json.load(f)
            if len(t_list) == 13:
                self.thresholds = torch.tensor(t_list, dtype=torch.float, device=self.device)
                print(f"--- Loaded {len(t_list)} thresholds from {thresholds_path} ---")
            else:
                print(f"--- Warning: Expected 13 thresholds, got {len(t_list)} ---")

    def setup(self, stage: str = None):
        if stage == 'fit':
            self.model.train()

    def on_train_epoch_start(self):
        self.model.train()

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

    @staticmethod
    def _mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        MInT (Mean-in-Transformer) pooling.
        Averages the last hidden states of all non-padding tokens,
        weighting each token equally (0 for padding, 1 for real tokens).
        """
        # Expand mask to match hidden-state shape [B, T, H]
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        # Avoid division by zero for (degenerate) all-padding sequences
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask):
        # Run the encoder
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Apply MInT: mean of all non-padding token last hidden states
        pooled = self._mean_pooling(outputs.last_hidden_state, attention_mask)
        # Project to 13 emotion logits
        return self.classifier(pooled)

    def calculate_loss(self, logits_13, targets_13):
        loss = self.loss_fn(logits_13, targets_13)
        return loss, targets_13

    def training_step(self, batch, batch_nb):
        inputs, targets = batch
        input_ids, attention_mask = self._safe_squeeze(inputs)

        logits_13 = self.forward(input_ids, attention_mask)
        loss, _ = self.calculate_loss(logits_13, targets['labels_aux'])

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.training_step_outputs.append({"loss": loss})
        return loss

    def validation_step(self, batch, batch_nb):
        inputs, targets = batch
        input_ids, attention_mask = self._safe_squeeze(inputs)

        logits_13 = self.forward(input_ids, attention_mask)
        loss, targets_13 = self.calculate_loss(logits_13, targets['labels_aux'])

        # Use custom thresholds if available
        probs_13 = torch.sigmoid(logits_13)
        preds_13 = (probs_13 > self.thresholds).float()

        output = {
            "val_loss": loss,
            "preds": preds_13,
            "targets": targets_13,
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

        per_class_jaccard = jaccard_score(
            all_targets_np,
            all_preds_np,
            average=None,
            zero_division=0
        )

        # 2. Log each label individually
        for i, score in enumerate(per_class_jaccard):
            # Using a prefix like 'val_class_jaccard/' groups them in TensorBoard/WandB
            self.log(f"val_class_jaccard/{i}", score, sync_dist=True)


        self.log("val_loss", avg_loss, prog_bar=True, sync_dist=True)
        self.log("val_jaccard", global_jaccard, prog_bar=True, sync_dist=True)

        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_nb):
        # FIX: Added test_step so trainer.test() doesn't crash or silently reuse
        # val data. Mirrors validation_step logic.
        inputs, targets = batch
        input_ids, attention_mask = self._safe_squeeze(inputs)

        logits_13 = self.forward(input_ids, attention_mask)
        loss, targets_13 = self.calculate_loss(logits_13, targets['labels_aux'])

        # Use custom thresholds if available
        probs_13 = torch.sigmoid(logits_13)
        preds_13 = (probs_13 > self.thresholds).float()

        output = {
            "test_loss": loss,
            "preds": preds_13,
            "targets": targets_13,
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
        # ASL hyperparameters
        parser.add_argument("--use_asl", default=True, type=bool, help="Use Asymmetric Loss instead of Focal Loss")
        parser.add_argument("--asl_gamma_neg", default=4.0, type=float)
        parser.add_argument("--asl_gamma_pos", default=1.0, type=float)
        parser.add_argument("--asl_clip", default=0.05, type=float)
        
        parser.add_argument("--encoder_model", default="SamLowe/roberta-base-go_emotions", type=str)
        parser.add_argument("--encoder_learning_rate", default=2e-5, type=float)
        parser.add_argument("--warmup_proportion", default=0.1, type=float)
        parser.add_argument("--max_length", default=128, type=int)
        parser.add_argument("--loader_workers", default=0, type=int)
        parser.add_argument("--train_csv", default="data/UsVsThem_train_public.csv", type=str)
        parser.add_argument("--dev_csv", default="data/UsVsThem_valid_public.csv", type=str)
        parser.add_argument("--test_csv", default="data/UsVsThem_test_public.csv", type=str)
        # Focal Loss hyperparameters
        parser.add_argument("--focal_gamma", default=2.0, type=float,
                            help="Focusing parameter for FocalLoss. 0 = standard BCE.")
        parser.add_argument("--focal_weight_clamp", default=20.0, type=float,
                            help="Max value for per-label pos_weight to avoid extreme gradients.")
        return parser