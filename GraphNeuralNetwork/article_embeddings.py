import argparse
import re

import numpy as np
import torch
import psycopg2
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import (
    StructType, StructField, StringType,
    ArrayType, FloatType
)
import pandas as pd
from transformers import AutoTokenizer


def get_args():
    parser = argparse.ArgumentParser(description='Embedd articles using RedditTransformer via Spark')

    parser.add_argument("--input_csv", type=str, required=True, help="Path to CSV with columns: article_id, body")
    parser.add_argument("--text_column", type=str, default="body", help="Column name containing article text")
    parser.add_argument("--id_column", type=str, default="article_id")

    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to BertRegression .ckpt file")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--num_groups", type=int, default=13, help="Number of emotion classes (aux head output size)")
    parser.add_argument("--num_classes", type=int, default=1, help="Main head output size")
    parser.add_argument("--extra_dropout", type=float, default=0.0)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--db_host", type=str, default="localhost")
    parser.add_argument("--db_port", type=int, default=5432)
    parser.add_argument("--db_name", type=str, required=True)
    parser.add_argument("--db_user", type=str, required=True)
    parser.add_argument("--db_password", type=str, required=True)

    # Spark
    parser.add_argument("--spark_partitions", type=int, default=8, help="Number of Spark partitions")

    return parser.parse_args()



_model_cache = {}
_tokenizer_cache = {}

def _get_model_and_tokenizer(checkpoint_path,model_name,num_groups,num_classes,extra_dropout):

    global _tokenizer_cache,model_cache

    key = checkpoint_path


    if key not in _model_cache:
        from SentimentClassification.RedditTransformer import RedditTransformer

        ckpt = torch.load(checkpoint_path, map_location = "cpu")

        state_dict = {
            k.replace("model.","",1):v
            for k,v in ckpt["state_dict"].items()
            if k.startswith("model.")
        }

        model = RedditTransformer(
            model_name=model_name,
            num_classes=num_classes,
            extra_dropout=extra_dropout,
            num_groups=num_groups,
        )

        model.load_state_dict(state_dict, strict=False)
        model.eval()
        _model_cache[key] = model
        _tokenizer_cache[key] = AutoTokenizer.from_pretrained(model_name)

def _clean_text(text: str) -> str:
    """Remove URLs (same preprocessing as MyCollator in dataloader.py)."""
    return re.sub(
        r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*',
        'LINK', str(text), flags=re.MULTILINE
    )

def embed_partition(
    partition_iter,
    checkpoint_path: str,
    model_name: str,
    num_classes: int,
    extra_dropout: float,
    num_groups: int,
    max_length: int,
    batch_size: int,
    conn_params: dict,
    id_col: str,
    text_col: str,
):
    """
    Called once per Spark partition. Iterates over rows in mini-batches,
    runs inference, and writes results directly to PostgreSQL.
    """
    model, tokenizer = _get_model_and_tokenizer(
        checkpoint_path, model_name, num_classes, extra_dropout, num_groups
    )

    buffer = []

    def flush(batch_rows):

        ids = [r[id_col] for r in batch_rows]
        texts = [_clean_text(r[text_col]) for r in batch_rows]

        tokenized = tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=max_length,
            stride=50,  # Overlap chunks by 50 tokens so we don't cut context in half
            return_overflowing_tokens=True,
            return_tensors="pt",
            add_special_tokens=True,
        )

        mapping = tokenized.pop("overflow_to_sample_mapping")

        with torch.no_grad():
            logits_main, logits_aux,hidden_states = model(tokenized.data)
            if isinstance(hidden_states, tuple):
                # we use the us vs them main task embeddings because it can detect bias and hostility better than just sentiments
                cls_emb = hidden_states[0][:, 0, :].cpu().numpy()  # [B, 768]
            else:
                cls_emb = hidden_states[:, 0, :].cpu().numpy()  # [B, 768]

                # --- Emotion vector from aux head (sigmoid for multi-label) ---
            if logits_aux is not None:
                emotion_vec = torch.sigmoid(logits_aux).cpu().numpy()  # [B, 13]
            else:
                emotion_vec = np.zeros((len(ids), num_groups), dtype=np.float32)

