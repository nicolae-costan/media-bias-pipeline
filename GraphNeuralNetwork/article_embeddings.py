"""
article_embeddings.py
---------------------
Reads articles from a CSV, runs them through the EmotionModel to produce:
  - A 768-dimensional embedding (mean pooling over the last hidden state)
  - 13 emotion probability scores

Writes results into two PostgreSQL tables:
  - articles           (article_id, body, outlet, topic, type, label_bias, news_link)
  - article_embeddings (article_id, embedding VECTOR(768), emotion_scores FLOAT4[])

Requires pgvector extension and the schema created by setup_db.py.

Usage:
    python GraphNeuralNetwork/article_embeddings.py \
        --input_csv "./data/merged_clean_data.csv" \
        --babe_csv "./data/final_labels_MBIC.csv" \
        --model_checkpoint "tb_logs/emotion_classification/version_0/checkpoints/epoch=2-val_loss=0.0853.ckpt"
"""

import argparse
import re
import os
import sys
import json
import pandas as pd
import numpy as np
import torch
import psycopg2
import psycopg2.extras
from tqdm import tqdm
from dotenv import load_dotenv

# Load .env from the same directory as this script
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)

# Add project root to system path to import utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils import compute_agreement

# Caches so the model is loaded only once
_model_cache = {}
_tokenizer_cache = {}


def get_args():
    parser = argparse.ArgumentParser(description="Embed articles and store in PostgreSQL with pgvector")

    # Input data
    parser.add_argument("--input_csv",  type=str, default="./data/merged_clean_data.csv",
                        help="CSV with article_id, body columns")
    parser.add_argument("--babe_csv",   type=str, default="./data/final_labels_MBIC.csv",
                        help="BABE CSV with outlet, topic, type, news_link metadata")
    parser.add_argument("--sg1_csv",    type=str, default="./data/raw_labels_SG1.csv")
    parser.add_argument("--sg2_csv",    type=str, default="./data/raw_labels_SG2.csv")
    
    parser.add_argument("--text_column", type=str, default="body")
    parser.add_argument("--id_column",   type=str, default="article_id")

    # Model
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--max_length",  type=int, default=512)
    parser.add_argument("--batch_size",  type=int, default=16)

    # Database (falls back to .env values)
    parser.add_argument("--db_host",     type=str, default=os.getenv("DB_HOST", "localhost"))
    parser.add_argument("--db_port",     type=int, default=int(os.getenv("DB_PORT", 5433)))
    parser.add_argument("--db_name",     type=str, default=os.getenv("DB_NAME", "media_bias"))
    parser.add_argument("--db_user",     type=str, default=os.getenv("DB_USER", "postgres"))
    parser.add_argument("--db_password", type=str, default=os.getenv("DB_PASSWORD", ""))

    return parser.parse_args()


def get_conn(args) -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host=args.db_host,
        port=args.db_port,
        dbname=args.db_name,
        user=args.db_user,
        password=args.db_password,
    )


def load_model_and_tokenizer(checkpoint_path: str):
    """Load (and cache) the EmotionModel and its tokenizer."""
    global _model_cache, _tokenizer_cache

    if checkpoint_path not in _model_cache:
        print(f"--- Loading model from {checkpoint_path} ---")

        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        emotion_dir = os.path.join(parent_dir, "EmotionModels")
        if parent_dir not in sys.path:
            sys.path.append(parent_dir)
        if emotion_dir not in sys.path:
            sys.path.append(emotion_dir)

        from EmotionModels.model import EmotionModel
        from transformers import AutoTokenizer

        model = EmotionModel.load_from_checkpoint(checkpoint_path, map_location="cpu")
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model.hparams.encoder_model)

        _model_cache[checkpoint_path] = model
        _tokenizer_cache[checkpoint_path] = tokenizer

    return _model_cache[checkpoint_path], _tokenizer_cache[checkpoint_path]


def _clean_text(text: str) -> str:
    """Replace URLs with the LINK token."""
    if not isinstance(text, str):
        return ""
    return re.sub(
        r"\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*",
        "LINK", text, flags=re.MULTILINE
    )


def load_and_merge_data(args) -> pd.DataFrame:
    """
    1. Load merged_clean_data.csv (article_id, body)
    2. Join with Consensus labels (Majority Voting from SG1/SG2)
    3. Join with Metadata (outlet, topic, etc. from BABE)
    """
    print(f"--- Reading {args.input_csv} ---")
    df_main = pd.read_csv(args.input_csv).dropna(subset=[args.id_column, args.text_column])
    df_main[args.id_column] = df_main[args.id_column].astype(str)
    
    # Drop existing label column from main if it exists, as we will re-calculate it
    if "label_bias" in df_main.columns:
        df_main = df_main.drop(columns=["label_bias"])

    # A. Calculate consensus labels
    if os.path.exists(args.sg1_csv) and os.path.exists(args.sg2_csv):
        df_consensus = compute_agreement(args.sg1_csv, args.sg2_csv, "article_id", "label_bias")
        df_main = df_main.merge(df_consensus, on="article_id", how="left")
        df_main = df_main.rename(columns={"consensus_label": "label_bias"})
    else:
        print("WARNING: SG1 or SG2 files missing. Cannot compute consensus labels.")
        df_main["label_bias"] = None
        df_main["agreement"] = 0.0

    # B. Get metadata from BABE
    if os.path.exists(args.babe_csv):
        print(f"--- Reading BABE metadata from {args.babe_csv} ---")
        df_babe = pd.read_csv(args.babe_csv, sep=";", on_bad_lines="skip")
        df_babe["article_id"] = df_babe["article_id"].astype(str)
        babe_meta = df_babe[["article_id", "outlet", "topic", "type", "news_link"]].drop_duplicates("article_id")
        df_main = df_main.merge(babe_meta, on="article_id", how="left")
    
    return df_main


def upsert_batch(cur, df_batch: pd.DataFrame, embeddings: np.ndarray, emotions: np.ndarray, id_col: str, text_col: str):
    """
    Upsert one batch into both tables.
    """
    # --- articles table ---
    article_rows = [
        (
            str(row[id_col]),
            row.get(text_col, None),
            row.get("outlet", None),
            row.get("topic", None),
            row.get("type", None),
            row.get("label_bias", None),
            row.get("news_link", None),
            float(row.get("agreement", 0.0))
        )
        for _, row in df_batch.iterrows()
    ]
    psycopg2.extras.execute_values(
        cur,
        """
        INSERT INTO articles (article_id, body, outlet, topic, type, label_bias, news_link, agreement)
        VALUES %s
        ON CONFLICT (article_id) DO UPDATE 
            SET label_bias = EXCLUDED.label_bias,
                agreement  = EXCLUDED.agreement
        """,
        article_rows,
    )

    # --- article_embeddings table ---
    embedding_rows = [
        (
            str(row[id_col]),
            embeddings[i].tolist(),          # Python list → pgvector accepts this
            emotions[i].tolist(),            # FLOAT4[]
        )
        for i, (_, row) in enumerate(df_batch.iterrows())
    ]
    psycopg2.extras.execute_values(
        cur,
        """
        INSERT INTO article_embeddings (article_id, embedding, emotion_scores)
        VALUES %s
        ON CONFLICT (article_id) DO UPDATE
            SET embedding      = EXCLUDED.embedding,
                emotion_scores = EXCLUDED.emotion_scores
        """,
        embedding_rows,
        template="(%s, %s::vector, %s::float4[])",
    )


def main():
    args = get_args()

    # 1. Load and merge data
    df = load_and_merge_data(args)
    total = len(df)

    # 2. Load model
    model, tokenizer = load_model_and_tokenizer(args.model_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Running inference on: {device}")

    # 3. Connect to Postgres
    print(f"--- Connecting to PostgreSQL at {args.db_host}:{args.db_port}/{args.db_name} ---")
    try:
        conn = get_conn(args)
    except psycopg2.OperationalError as e:
        print(f"\nERROR: Could not connect to the database.\n{e}")
        print("\nRun setup_db.py first, and make sure the Docker container is running:")
        print("  docker start media-bias-postgres")
        sys.exit(1)

    conn.autocommit = False
    cur = conn.cursor()

    # 4. Process in batches
    print(f"--- Starting embedding process (batch_size={args.batch_size}) ---")
    for start_idx in tqdm(range(0, total, args.batch_size)):
        end_idx = min(start_idx + args.batch_size, total)
        batch_df = df.iloc[start_idx:end_idx]

        texts = [_clean_text(t) for t in batch_df[args.text_column].tolist()]

        # Tokenize
        tokenized = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model.model(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                output_hidden_states=True,
            )
            # Mean pooling over the last hidden state (MInT)
            last_hidden = outputs.hidden_states[-1]
            mask = tokenized["attention_mask"].unsqueeze(-1)
            sum_hidden = (last_hidden * mask).sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1e-9)
            embeddings = (sum_hidden / count).cpu().numpy()   # [B, 768]

            # Emotion scores via the classifier head
            logits_aux = model.classifier(sum_hidden / count)
            emotions = torch.sigmoid(logits_aux).cpu().numpy()  # [B, 13]

        # Write to Postgres
        upsert_batch(cur, batch_df, embeddings, emotions, args.id_column, args.text_column)
        conn.commit()

    cur.close()
    conn.close()
    print(f"\n--- SUCCESS! {total:,} articles embedded and stored in PostgreSQL ---")


if __name__ == "__main__":
    main()