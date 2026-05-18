import math
import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AutoModel, AutoConfig
from sklearn.metrics import jaccard_score, f1_score
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
        probs = torch.sigmoid(logits)

        # Positive samples
        loss_pos = targets * torch.log(probs.clamp(min=self.eps))
        if self.gamma_pos > 0:
            loss_pos *= (1 - probs) ** self.gamma_pos

        # Apply pos_weight only to the positive term
        if self.pos_weight is not None:
            loss_pos = loss_pos * self.pos_weight

        # Negative samples
        probs_neg = 1 - probs
        if self.clip is not None and self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1)

        loss_neg = (1 - targets) * torch.log(probs_neg.clamp(min=self.eps))
        if self.gamma_neg > 0:
            loss_neg *= (1 - probs_neg) ** self.gamma_neg

        loss = -(loss_pos + loss_neg)
        return loss.mean()


def compute_weights_from_csv(csv_paths, label_cols, clamp_max=20.0):
    """Compute per-label positive-class weights from one or more CSV files.

    Args:
        csv_paths  : list of paths to CSV files that contain the label columns.
        label_cols : list of column names to treat as binary labels.
        clamp_max  : maximum allowed weight (prevents extreme gradient spikes).

    Returns:
        (weight_tensor, total_rows, label_cols)
    """

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

    df_all = pd.concat(dfs, ignore_index=True)
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

        # Derive the number of output classes from the dataset definition so the
        # model works with any label set (7 for combined, 13 for UsVsThem, etc.)
        from dataloader import RedditDataset
        num_emotions = len(RedditDataset.EMOTION_COLUMNS)
        self.num_emotions = num_emotions

        # Fresh N-class linear head
        self.classifier = nn.Linear(hidden_size, num_emotions)

        pos_weight, _, _ = compute_weights_from_csv(
            [self.hparams.train_csv, self.hparams.dev_csv],
            label_cols=RedditDataset.EMOTION_COLUMNS,
            clamp_max=getattr(self.hparams, 'focal_weight_clamp', 20.0),
        )

        if getattr(self.hparams, 'use_asl', True):
            self.loss_fn = AsymmetricLoss(
                gamma_neg=getattr(self.hparams, 'asl_gamma_neg', 4.0),
                gamma_pos=getattr(self.hparams, 'asl_gamma_pos', 1.0),
                clip=getattr(self.hparams, 'asl_clip', 0.05),
                pos_weight=pos_weight
            )
        else:
            self.loss_fn = FocalLoss(
                gamma=getattr(self.hparams, 'focal_gamma', 2.0),
                pos_weight=pos_weight,
                reduction='mean',
            )

        # Default thresholds (0.5 for all emotion nodes)
        self.register_buffer('thresholds', torch.ones(num_emotions, dtype=torch.float) * 0.5)

        # Freeze the encoder for the first `freeze_epochs` epochs so the
        # classifier head can warm-start before full fine-tuning begins.
        self._freeze_encoder_backbone()

    def load_thresholds(self, thresholds_path):
        if os.path.exists(thresholds_path):
            with open(thresholds_path, 'r') as f:
                t_list = json.load(f)
            if len(t_list) == self.num_emotions:
                self.thresholds = torch.tensor(t_list, dtype=torch.float, device=self.device)
                print(f"--- Loaded {len(t_list)} thresholds from {thresholds_path} ---")
            else:
                print(f"--- Warning: Expected {self.num_emotions} thresholds, got {len(t_list)}. Using defaults. ---")

    # ------------------------------------------------------------------ #
    #  Freeze / unfreeze helpers                                          #
    # ------------------------------------------------------------------ #

    def _freeze_encoder_backbone(self):
        """Freeze all encoder (RoBERTa) parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        log.info("[freeze] Encoder frozen — only classifier head is trainable.")

    def _unfreeze_encoder_backbone(self):
        """Unfreeze all encoder parameters so the whole model fine-tunes."""
        for param in self.model.parameters():
            param.requires_grad = True
        log.info("[freeze] Encoder unfrozen — full model is now trainable.")

    def setup(self, stage: str = None):
        if stage == 'fit':
            self.model.train()

    def on_train_epoch_start(self):
        """Unfreeze the encoder once we have completed `freeze_epochs` epochs."""
        freeze_epochs = getattr(self.hparams, 'freeze_epochs', 0)
        if freeze_epochs > 0 and self.current_epoch == freeze_epochs:
            self._unfreeze_encoder_backbone()
            log.info(
                f"[freeze] Epoch {self.current_epoch}: encoder unfrozen after "
                f"{freeze_epochs} warm-up epoch(s)."
            )
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
        # Project to N emotion logits
        return self.classifier(pooled)

    def calculate_loss(self, logits, targets):
        loss = self.loss_fn(logits, targets)
        return loss, targets

    def training_step(self, batch, batch_nb):
        inputs, targets = batch
        input_ids, attention_mask = self._safe_squeeze(inputs)

        logits = self.forward(input_ids, attention_mask)
        loss, _ = self.calculate_loss(logits, targets['labels_aux'])

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.training_step_outputs.append({"loss": loss})
        return loss

    def validation_step(self, batch, batch_nb):
        inputs, targets = batch
        input_ids, attention_mask = self._safe_squeeze(inputs)

        logits = self.forward(input_ids, attention_mask)
        loss, targets_out = self.calculate_loss(logits, targets['labels_aux'])

        # Use custom thresholds if available
        probs = torch.sigmoid(logits)
        preds = (probs > self.thresholds).float()

        output = {
            "val_loss": loss,
            "preds": preds,
            "targets": targets_out,
        }
        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()

        all_preds   = torch.cat([x["preds"]   for x in self.validation_step_outputs], dim=0)
        all_targets = torch.cat([x["targets"] for x in self.validation_step_outputs], dim=0)

        all_preds_np   = all_preds.cpu().numpy()
        all_targets_np = all_targets.cpu().numpy()

        from dataloader import RedditDataset
        emotion_names = RedditDataset.EMOTION_COLUMNS

        # ── macro metrics ──────────────────────────────────────────────────
        global_jaccard = jaccard_score(
            all_targets_np, all_preds_np, average="macro", zero_division=0
        )
        global_f1 = f1_score(
            all_targets_np, all_preds_np, average="macro", zero_division=0
        )

        # ── per-class metrics ──────────────────────────────────────────────
        per_class_jaccard = jaccard_score(
            all_targets_np, all_preds_np, average=None, zero_division=0
        )
        per_class_f1 = f1_score(
            all_targets_np, all_preds_np, average=None, zero_division=0
        )

        # Log each class under a named key (groups nicely in TensorBoard)
        for i, name in enumerate(emotion_names):
            self.log(f"val_class_jaccard/{name}", per_class_jaccard[i], sync_dist=True)
            self.log(f"val_class_f1/{name}",      per_class_f1[i],      sync_dist=True)

        # ── console table (visible in training output) ─────────────────────
        header = f"  {'Emotion':<14} {'Jaccard':>8}  {'F1':>8}"
        print(f"\n  [Epoch {self.current_epoch}] Validation per-emotion metrics:")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for i, name in enumerate(emotion_names):
            print(f"  {name:<14} {per_class_jaccard[i]:>8.4f}  {per_class_f1[i]:>8.4f}")
        print("  " + "-" * (len(header) - 2))
        print(f"  {'MACRO':14} {global_jaccard:>8.4f}  {global_f1:>8.4f}\n")

        self.log("val_loss",    avg_loss,       prog_bar=True, sync_dist=True)
        self.log("val_jaccard", global_jaccard, prog_bar=True, sync_dist=True)
        self.log("val_f1",      global_f1,      prog_bar=True, sync_dist=True)

        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_nb):
        # Added test_step so trainer.test() doesn't crash or silently reuse
        # val data. Mirrors validation_step logic.
        inputs, targets = batch
        input_ids, attention_mask = self._safe_squeeze(inputs)

        logits = self.forward(input_ids, attention_mask)
        loss, targets_out = self.calculate_loss(logits, targets['labels_aux'])

        # Use custom thresholds if available
        probs = torch.sigmoid(logits)
        preds = (probs > self.thresholds).float()

        output = {
            "test_loss": loss,
            "preds": preds,
            "targets": targets_out,
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

    # ------------------------------------------------------------------ #
    #  Optimizer helpers (layer-wise LR + decay / no-decay split)         #
    # ------------------------------------------------------------------ #

    _NO_DECAY_SUFFIXES = ("bias", "LayerNorm.weight")

    @classmethod
    def _param_no_weight_decay(cls, param_name: str) -> bool:
        return any(nd in param_name for nd in cls._NO_DECAY_SUFFIXES)

    def _append_lr_groups(
        self,
        optimizer_groups: list,
        named_params,
        lr: float,
        weight_decay: float,
    ) -> None:
        """Split a module's parameters into decay / no-decay AdamW groups."""
        decay_params, no_decay_params = [], []
        for name, param in named_params:
            if self._param_no_weight_decay(name):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        if decay_params:
            optimizer_groups.append(
                {"params": decay_params, "lr": lr, "weight_decay": weight_decay}
            )
        if no_decay_params:
            optimizer_groups.append(
                {"params": no_decay_params, "lr": lr, "weight_decay": 0.0}
            )

    def _encoder_layer_modules(self):
        """Return (embeddings_module, list_of_transformer_layers) or (None, None)."""
        if hasattr(self.model, "embeddings") and hasattr(self.model, "encoder"):
            encoder = self.model.encoder
            if hasattr(encoder, "layer"):
                return self.model.embeddings, list(encoder.layer)
        return None, None

    def _build_optimizer_param_groups(self) -> list:
        base_lr = self.hparams.encoder_learning_rate
        weight_decay = getattr(self.hparams, "weight_decay", 0.01)
        head_mult = getattr(self.hparams, "head_lr_multiplier", 10.0)
        layer_decay = getattr(self.hparams, "layerwise_lr_decay", 0.9)

        optimizer_grouped: list = []
        embeddings, layers = self._encoder_layer_modules()

        if layers is not None:
            num_layers = len(layers)
            # Embeddings sit below layer 0 — lowest LR when layer_decay < 1
            emb_lr = base_lr * (layer_decay ** num_layers)
            self._append_lr_groups(
                optimizer_grouped,
                embeddings.named_parameters(),
                emb_lr,
                weight_decay,
            )
            # Deeper transformer blocks get progressively larger LRs
            for i, layer in enumerate(layers):
                layer_lr = base_lr * (layer_decay ** (num_layers - 1 - i))
                self._append_lr_groups(
                    optimizer_grouped,
                    layer.named_parameters(),
                    layer_lr,
                    weight_decay,
                )
        else:
            # Non–layer-stacked encoders (fallback): single encoder LR
            self._append_lr_groups(
                optimizer_grouped,
                self.model.named_parameters(),
                base_lr,
                weight_decay,
            )

        # Classification head — highest LR
        self._append_lr_groups(
            optimizer_grouped,
            self.classifier.named_parameters(),
            base_lr * head_mult,
            weight_decay,
        )
        return optimizer_grouped

    def configure_optimizers(self):
        optimizer = optim.AdamW(self._build_optimizer_param_groups())

        # Compute train steps directly from the dataset length instead of
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
        parser.add_argument(
            "--freeze_epochs", default=1, type=int,
            help="Number of epochs to keep the encoder frozen (0 = never freeze). "
                 "During these epochs only the classifier head is trained."
        )
        parser.add_argument("--encoder_learning_rate", default=2e-5, type=float)
        parser.add_argument(
            "--weight_decay", default=0.01, type=float,
            help="AdamW weight decay for non–bias / non–LayerNorm parameters.",
        )
        parser.add_argument(
            "--head_lr_multiplier", default=10.0, type=float,
            help="Classifier head LR = encoder_learning_rate × this factor.",
        )
        parser.add_argument(
            "--layerwise_lr_decay", default=0.9, type=float,
            help="Per-layer LR multiplier (<1 raises LR toward the top of the stack). "
                 "Set to 1.0 for a uniform encoder LR.",
        )
        parser.add_argument("--warmup_proportion", default=0.1, type=float)
        parser.add_argument("--max_length", default=128, type=int)
        parser.add_argument("--loader_workers", default=0, type=int)
        parser.add_argument("--train_csv", default="data/combined_train_dataset.csv", type=str)
        parser.add_argument("--dev_csv", default="data/combined_valid_dataset.csv", type=str)
        parser.add_argument("--test_csv", default="data/combined_test_dataset.csv", type=str)
        # Focal Loss hyperparameters
        parser.add_argument("--focal_gamma", default=2.0, type=float,
                            help="Focusing parameter for FocalLoss. 0 = standard BCE.")
        parser.add_argument("--focal_weight_clamp", default=20.0, type=float,
                            help="Max value for per-label pos_weight to avoid extreme gradients.")
        return parser