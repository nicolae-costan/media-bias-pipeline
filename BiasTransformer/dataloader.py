import re
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


LABEL_TO_ID = {"Non-biased": 0, "Biased": 1}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


def _resolve_csv_path(csv_path: str) -> str:
    path = Path(csv_path)
    if path.exists():
        return str(path)

    project_root = Path(__file__).resolve().parents[1]
    root_path = project_root / csv_path
    if root_path.exists():
        return str(root_path)

    raise FileNotFoundError(f"CSV file not found: {csv_path}")


class BiasDataset(Dataset):
    def __init__(self, csv_path: str):
        csv_path = _resolve_csv_path(csv_path)
        self.df = pd.read_csv(csv_path)
        self.df = self.df.dropna(subset=["body", "label"]).reset_index(drop=True)
        self.df["label"] = self.df["label"].map(LABEL_TO_ID).astype(int)

        if "sample_weight" not in self.df.columns:
            self.df["sample_weight"] = 1.0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "article_id": row["article_id"],
            "body": row["body"],
            "label": int(row["label"]),
            "sample_weight": float(row["sample_weight"]),
        }


class BiasCollator:
    def __init__(self, model_name: str, max_length: int):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def __call__(self, batch):
        texts = [
            re.sub(
                r"\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*",
                "LINK",
                str(item["body"]),
                flags=re.MULTILINE,
            )
            for item in batch
        ]

        tokenized = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )

        targets = {
            "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
            "sample_weight": torch.tensor([item["sample_weight"] for item in batch], dtype=torch.float),
            "article_id": [item["article_id"] for item in batch],
        }
        return tokenized.data, targets
