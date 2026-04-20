"""
Dataset & Collator for the "Us vs. Them" Reddit Bias Dataset
=============================================================
Handles all 3 label sets required by Module A:
  - Bias score    (float, 0.0–1.0)
  - Emotions      (13-dim binary vector)
  - Social group  (categorical string → integer index)
"""

import re
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader, RandomSampler
from transformers import AutoTokenizer

log = logging.getLogger(__name__)

# ── Emotion column names (must match CSV headers exactly) ───────────────
EMOTION_COLUMNS: List[str] = [
    "Anger", "Contempt", "Disgust", "Fear", "Gratitude", "Guilt",
    "Happiness", "Hope", "Pride", "Relief", "Sadness", "Sympathy",
    "Emotions_Neutral",
]

# Regex to replace URLs with a placeholder token
_URL_PATTERN = re.compile(
    r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*',
    flags=re.MULTILINE,
)


class UsVsThemDataset(Dataset):
    """
    Loads one split (train/val/test) of the Us vs. Them dataset.

    Each sample yields:
        - text:           raw comment body (str)
        - bias_score:     continuous target ∈ [0, 1] (float)
        - emotion_vector: 13-dim binary vector (List[int])
        - social_group:   integer-encoded group label (int)
    """

    def __init__(
        self,
        csv_path: str,
        group_encoder: LabelEncoder,
    ) -> None:
        super().__init__()
        self.df = pd.read_csv(csv_path)

        # ── Validate required columns ───────────────────────────────────
        required = {"body", "usVSthem_scale", "group"} | set(EMOTION_COLUMNS)
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(
                f"CSV {csv_path} is missing columns: {missing}"
            )

        # ── Pre-compute emotion vectors as lists ────────────────────────
        self.emotions: List[List[int]] = (
            self.df[EMOTION_COLUMNS].values.tolist()
        )

        # ── Encode social group labels ──────────────────────────────────
        self.group_labels: List[int] = (
            group_encoder.transform(self.df["group"].values).tolist()
        )

        log.info(
            f"Loaded {len(self.df)} samples from {csv_path} "
            f"({len(group_encoder.classes_)} groups)"
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        return {
            "text":           str(row["body"]),
            "bias_score":     float(row["usVSthem_scale"]),
            "emotion_vector": self.emotions[idx],
            "social_group":   self.group_labels[idx],
        }


class MTLCollator:
    """
    Tokenizes text batches and packages labels for all 3 heads.

    Returns:
        Tuple of (inputs_dict, labels_dict) where:
            inputs_dict: {"input_ids": [B,S], "attention_mask": [B,S]}
            labels_dict: {"labels_bias": [B], "labels_emotion": [B,13],
                          "labels_social": [B]}
    """

    def __init__(self, model_name: str, max_length: int = 512) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def __call__(
        self, batch: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

        # ── Clean text (replace URLs) ───────────────────────────────────
        texts = [
            _URL_PATTERN.sub("LINK", sample["text"]) for sample in batch
        ]

        # ── Tokenize ────────────────────────────────────────────────────
        encoded = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )

        # ── Package labels ──────────────────────────────────────────────
        labels = {
            "labels_bias": torch.tensor(
                [s["bias_score"] for s in batch], dtype=torch.float
            ),
            "labels_emotion": torch.tensor(
                [s["emotion_vector"] for s in batch], dtype=torch.float
            ),
            "labels_social": torch.tensor(
                [s["social_group"] for s in batch], dtype=torch.long
            ),
        }

        return encoded.data, labels


def compute_emotion_pos_weights(train_csv: str) -> torch.Tensor:
    """
    Compute per-emotion positive class weights from the training set.

    Weight = num_negatives / num_positives for each emotion column.
    This counteracts class imbalance by making rare positive emotions
    contribute more to the loss.

    Returns:
        Tensor of shape [13] with per-emotion positive weights.
    """
    df = pd.read_csv(train_csv)
    emotion_data = df[EMOTION_COLUMNS].values.astype(float)
    pos_counts = emotion_data.sum(axis=0)  # [13]
    neg_counts = len(df) - pos_counts
    # Clamp to avoid division by zero for always-positive columns
    weights = neg_counts / np.clip(pos_counts, a_min=1.0, a_max=None)
    # Cap at 10x to prevent extreme weights
    weights = np.clip(weights, a_min=1.0, a_max=10.0)
    log.info(f"Emotion pos_weights: {dict(zip(EMOTION_COLUMNS, weights.round(2)))}")
    return torch.tensor(weights, dtype=torch.float)


def build_dataloaders(
    train_csv: str,
    dev_csv: str,
    test_csv: str,
    model_name: str = "roberta-base",
    batch_size: int = 16,
    max_length: int = 512,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader, LabelEncoder, torch.Tensor]:
    """
    Build train/val/test DataLoaders with a shared LabelEncoder.

    The LabelEncoder is fit on the union of all splits to ensure
    consistent group-label indexing across train/val/test.

    Returns:
        (train_loader, val_loader, test_loader, group_encoder, emotion_pos_weights)
    """
    # ── Fit LabelEncoder on ALL splits ──────────────────────────────────
    all_groups = pd.concat([
        pd.read_csv(train_csv)["group"],
        pd.read_csv(dev_csv)["group"],
        pd.read_csv(test_csv)["group"],
    ])
    group_encoder = LabelEncoder()
    group_encoder.fit(all_groups.values)

    num_groups = len(group_encoder.classes_)
    log.info(f"Social groups ({num_groups}): {list(group_encoder.classes_)}")

    # ── Compute emotion class weights from training data ────────────────
    emotion_pos_weights = compute_emotion_pos_weights(train_csv)

    # ── Build datasets ──────────────────────────────────────────────────
    train_ds = UsVsThemDataset(train_csv, group_encoder)
    val_ds   = UsVsThemDataset(dev_csv,   group_encoder)
    test_ds  = UsVsThemDataset(test_csv,  group_encoder)

    # ── Build collator ──────────────────────────────────────────────────
    collator = MTLCollator(model_name, max_length)

    # ── Build DataLoaders ───────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds,
        sampler=RandomSampler(train_ds),
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, group_encoder, emotion_pos_weights
