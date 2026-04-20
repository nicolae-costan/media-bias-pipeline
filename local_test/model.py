"""
Module A: Sentence-Level Bias & Emotion Engine
===============================================
RoBERTa-based Multi-Task Learning model with shared encoder layers 1–11
and 3 task-specific branches at layer 12.

Architecture:
    ┌─────────────────────────────────────────────┐
    │  RoBERTa Embeddings                         │
    │  Shared Transformer Layers 1–11             │
    ├──────────┬──────────────┬────────────────────┤
    │ Layer 12 │  Layer 12    │  Layer 12          │
    │ (Bias)   │  (Emotion)   │  (Social Group)    │
    ├──────────┼──────────────┼────────────────────┤
    │ MeanPool │  MeanPool    │  MeanPool          │
    ├──────────┼──────────────┼────────────────────┤
    │ Linear   │  Linear      │  Linear            │
    │ Tanh     │  Tanh        │  Tanh              │
    │ Dropout  │  Dropout     │  Dropout           │
    │ Linear→1 │  Linear→13   │  Linear→N_groups   │
    │ Sigmoid  │  (raw logits)│  (raw logits)      │
    └──────────┴──────────────┴────────────────────┘

Heads:
    1. Primary:   "Us vs. Them" bias score (Regression, 0.0–1.0)
    2. Auxiliary1: Emotion prediction (Multi-label classification)
    3. Auxiliary2: Targeted social group (Single-label classification)
"""

import copy
import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

log = logging.getLogger(__name__)

# ── Emotion column names (fixed by dataset schema) ──────────────────────
EMOTION_COLUMNS: List[str] = [
    "Anger", "Contempt", "Disgust", "Fear", "Gratitude", "Guilt",
    "Happiness", "Hope", "Pride", "Relief", "Sadness", "Sympathy",
    "Emotions_Neutral",
]
NUM_EMOTIONS: int = len(EMOTION_COLUMNS)  # 13


class RoBERTaMTL(nn.Module):
    """
    Multi-Task Learning RoBERTa model.

    Shared encoder (layers 1–11) feeds into 3 independent branches,
    each with its own copy of layer 12, mean-pooling, and task head.

    Args:
        model_name:        HuggingFace model identifier (default: roberta-base).
        num_emotions:      Number of emotion labels (default: 13).
        num_social_groups: Number of social group classes.
        extra_dropout:     Additional dropout applied on top of model defaults.
        loss_weights:      Per-task loss multipliers for the combined objective.
    """

    def __init__(
        self,
        model_name: str = "roberta-base",
        num_emotions: int = NUM_EMOTIONS,
        num_social_groups: int = 10,
        extra_dropout: float = 0.1,
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()

        # ── 1. Load pre-trained config & model ──────────────────────────
        config = AutoConfig.from_pretrained(model_name)
        if extra_dropout > 0:
            config.hidden_dropout_prob += extra_dropout
            config.attention_probs_dropout_prob += extra_dropout

        self.config = config
        self.loss_weights = loss_weights or {
            "bias": 1.0,
            "emotion": 1.0,
            "social": 1.0,
        }

        base_model = AutoModel.from_pretrained(model_name, config=config)
        self.encoder = base_model

        # ── 2. Branch layer 12 into 3 task-specific copies ──────────────
        layer_12 = self.encoder.encoder.layer[11]
        self.layer_bias    = copy.deepcopy(layer_12)
        self.layer_emotion = copy.deepcopy(layer_12)
        self.layer_social  = copy.deepcopy(layer_12)

        # ── 3. Truncate shared encoder to layers 1–11 ──────────────────
        self.encoder.encoder.layer = nn.ModuleList(
            self.encoder.encoder.layer[:11]
        )

        # ── 4. Task-specific classification heads ───────────────────────
        h = config.hidden_size
        drop = config.hidden_dropout_prob

        # Primary: Bias Score Regression (output ∈ [0, 1])
        self.bias_head = nn.Sequential(
            nn.Linear(h, h // 2),
            nn.Tanh(),
            nn.Dropout(drop),
            nn.Linear(h // 2, 1),
            nn.Sigmoid(),
        )

        # Auxiliary 1: Emotion Multi-Label (raw logits → BCEWithLogitsLoss)
        self.emotion_head = nn.Sequential(
            nn.Linear(h, h // 2),
            nn.Tanh(),
            nn.Dropout(drop),
            nn.Linear(h // 2, num_emotions),
        )

        # Auxiliary 2: Social Group Classification (raw logits → CrossEntropyLoss)
        self.social_head = nn.Sequential(
            nn.Linear(h, h // 2),
            nn.Tanh(),
            nn.Dropout(drop),
            nn.Linear(h // 2, num_social_groups),
        )

        # ── 5. Weight init for new components ───────────────────────────
        self._init_custom_weights()

        # ── 6. Ablation state ───────────────────────────────────────────
        self._ablate_social: bool = False
        self.num_emotions = num_emotions
        self.num_social_groups = num_social_groups

        log.info(
            f"RoBERTaMTL initialized: emotions={num_emotions}, "
            f"social_groups={num_social_groups}, ablate={self._ablate_social}"
        )

    # ── Weight Initialization ───────────────────────────────────────────

    def _init_custom_weights(self) -> None:
        """Apply the encoder's native initialization to all new layers."""
        for module in [self.layer_bias, self.layer_emotion, self.layer_social]:
            self.encoder._init_weights(module)

        for head in [self.bias_head, self.emotion_head, self.social_head]:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    self.encoder._init_weights(m)

    # ── Ablation Control ────────────────────────────────────────────────

    def set_ablation_mode(self, ablate: bool = True) -> None:
        """
        Enable/disable social group head for Phase 2 fine-tuning.

        When ablated:
          - Social head parameters are frozen (no gradient updates)
          - Social logits are zeroed out
          - Social loss is excluded from the combined objective
        """
        self._ablate_social = ablate

        for param in self.social_head.parameters():
            param.requires_grad = not ablate
        for param in self.layer_social.parameters():
            param.requires_grad = not ablate

        status = "ABLATED (frozen)" if ablate else "ACTIVE"
        log.info(f"Social group head: {status}")

    @property
    def is_ablated(self) -> bool:
        return self._ablate_social

    # ── Pooling ─────────────────────────────────────────────────────────

    @staticmethod
    def _mean_pooling(
        hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Average token embeddings while ignoring padding tokens.

        Args:
            hidden_states:  [B, S, H] last hidden states from a transformer layer.
            attention_mask: [B, S] binary mask (1 = real token, 0 = pad).

        Returns:
            Pooled tensor of shape [B, H].
        """
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        summed = torch.sum(hidden_states * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    # ── Forward Pass ────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels_bias: Optional[torch.Tensor] = None,
        labels_emotion: Optional[torch.Tensor] = None,
        labels_social: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through shared encoder + 3 task branches.

        Args:
            input_ids:      [B, S] token IDs.
            attention_mask: [B, S] attention mask.
            labels_bias:    [B] float targets for bias regression.
            labels_emotion: [B, 13] binary targets for emotion multi-label.
            labels_social:  [B] integer targets for social group classification.

        Returns:
            Dictionary with keys:
                - bias_score:    [B, 1] predicted bias scores
                - emotions:      [B, 13] emotion logits
                - social_group:  [B, N] social group logits
                - loss:          scalar combined loss (if any labels provided)
                - loss_bias, loss_emotion, loss_social: individual losses
        """
        # ── Shared layers 1–11 ──────────────────────────────────────────
        encoder_out = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        shared_hidden = encoder_out.last_hidden_state  # [B, S, H]

        # ── Extended mask for manual layer-12 calls ─────────────────────
        ext_mask = self.encoder.get_extended_attention_mask(
            attention_mask, input_ids.shape
        )

        # ── Task-specific layer-12 branches ─────────────────────────────
        def _run_branch(layer: nn.Module) -> torch.Tensor:
            out = layer(shared_hidden, attention_mask=ext_mask)
            return out[0] if isinstance(out, tuple) else out

        hidden_bias    = _run_branch(self.layer_bias)
        hidden_emotion = _run_branch(self.layer_emotion)
        hidden_social  = _run_branch(self.layer_social)

        # ── Mean-pool each branch ───────────────────────────────────────
        pooled_bias    = self._mean_pooling(hidden_bias, attention_mask)
        pooled_emotion = self._mean_pooling(hidden_emotion, attention_mask)
        pooled_social  = self._mean_pooling(hidden_social, attention_mask)

        # ── Classification heads ────────────────────────────────────────
        bias_score    = self.bias_head(pooled_bias)          # [B, 1]
        emotion_logits = self.emotion_head(pooled_emotion)   # [B, 13]
        social_logits  = self.social_head(pooled_social)     # [B, N]

        # Zero out social logits during ablation
        if self._ablate_social:
            social_logits = torch.zeros_like(social_logits)

        results: Dict[str, torch.Tensor] = {
            "bias_score":   bias_score,
            "emotions":     emotion_logits,
            "social_group": social_logits,
        }

        # ── Loss computation (only if labels are provided) ──────────────
        total_loss = torch.tensor(0.0, device=input_ids.device)
        has_loss = False

        if labels_bias is not None:
            loss_bias = nn.functional.mse_loss(
                bias_score.squeeze(-1), labels_bias
            )
            total_loss = total_loss + self.loss_weights["bias"] * loss_bias
            results["loss_bias"] = loss_bias
            has_loss = True

        if labels_emotion is not None:
            loss_emotion = nn.functional.binary_cross_entropy_with_logits(
                emotion_logits, labels_emotion
            )
            total_loss = total_loss + self.loss_weights["emotion"] * loss_emotion
            results["loss_emotion"] = loss_emotion
            has_loss = True

        if labels_social is not None and not self._ablate_social:
            loss_social = nn.functional.cross_entropy(
                social_logits, labels_social.long()
            )
            total_loss = total_loss + self.loss_weights["social"] * loss_social
            results["loss_social"] = loss_social
            has_loss = True

        if has_loss:
            results["loss"] = total_loss

        return results

    # ── Parameter Groups (for differential learning rates) ──────────────

    def get_shared_params(self) -> List[nn.Parameter]:
        """Parameters for shared layers 1–11 + embeddings."""
        return list(self.encoder.parameters())

    def get_head_params(self) -> List[nn.Parameter]:
        """Parameters for all 3 task-specific branches + heads."""
        params: List[nn.Parameter] = []
        for module in [
            self.layer_bias, self.layer_emotion, self.layer_social,
            self.bias_head, self.emotion_head, self.social_head,
        ]:
            params.extend(module.parameters())
        return params

    def get_active_head_params(self) -> List[nn.Parameter]:
        """Like get_head_params(), but excludes ablated social head."""
        params: List[nn.Parameter] = []
        for module in [self.layer_bias, self.layer_emotion,
                       self.bias_head, self.emotion_head]:
            params.extend(module.parameters())
        if not self._ablate_social:
            for module in [self.layer_social, self.social_head]:
                params.extend(module.parameters())
        return params
