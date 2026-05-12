import argparse
import re
import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

import numpy as np
import torch
import psycopg2
import psycopg2.extras
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from collections import defaultdict
from transformers import AutoTokenizer


_model_cache = {}
_tokenizer_cache = {}


def get_args():
    """
    Parse command-line arguments for the embedding pipeline.
    """
    parser = argparse.ArgumentParser(description='Embed articles using EmotionModel via Spark')

    parser.add_argument("--input_csv", type=str, default=os.getenv("INPUT_CSV", "../data/merged_clean_data.csv"))
    parser.add_argument("--text_column", type=str, default="body")
    parser.add_argument("--id_column", type=str, default="article_id")

    parser.add_argument("--model_checkpoint", type=str, default=os.getenv("MODEL_CHECKPOINT"))
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--num_groups", type=int, default=13)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--extra_dropout", type=float, default=0.0)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--driver_memory", type=str, default="2g")
    parser.add_argument("--executor_memory", type=str, default="3g")

    # DB — no WSL host auto-detection; use DB_HOST from .env or explicit arg
    parser.add_argument("--db_host", type=str, default=os.getenv("DB_HOST", "localhost"))
    parser.add_argument("--db_port", type=int, default=int(os.getenv("DB_PORT", 5433)))
    parser.add_argument("--db_name", type=str, default=os.getenv("DB_NAME", "media_bias"))
    parser.add_argument("--db_user", type=str, default=os.getenv("DB_USER", "postgres"))
    parser.add_argument("--db_password", type=str, default=os.getenv("DB_PASSWORD", ""))

    parser.add_argument("--spark_partitions", type=int, default=8)

    return parser.parse_args()


def _get_model_and_tokenizer(checkpoint_path, model_name, num_groups, num_classes, extra_dropout):
    """
          Load and cache the model and tokenizer for inference.

          Initializes a Emotion model from a checkpoint and pairs it with
          a HuggingFace tokenizer. Uses a global cache to avoid reloading per partition.

          Args:
              Reddit transformers parameters
              checkpoint_path (str): Path to the model checkpoint file.
              model_name (str): HuggingFace model name for tokenizer initialization.
              num_groups (int): Number of auxiliary emotion classes.
              num_classes (int): Number of main output classes.
              extra_dropout (float): Additional dropout applied in the model.

          Returns:
              Tuple[torch.nn.Module, transformers.PreTrainedTokenizer]:
                  Loaded model in evaluation mode and its tokenizer.

          Side Effects:
              - Loads model weights into memory.
              - Mutates global caches (_model_cache, _tokenizer_cache).
       """
    global _model_cache, _tokenizer_cache

    key = checkpoint_path
    if key not in _model_cache:
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        emotion_dir = os.path.join(parent_dir, 'EmotionModels')
        sys.path.append(parent_dir)
        sys.path.append(emotion_dir)

        from EmotionModels.model import EmotionModel
        import sklearn.preprocessing

        if hasattr(torch.serialization, 'add_safe_globals'):
            torch.serialization.add_safe_globals([sklearn.preprocessing.LabelEncoder])

        model = EmotionModel.load_from_checkpoint(checkpoint_path, map_location="cpu")
        model.eval()
        _model_cache[key] = model
        _tokenizer_cache[key] = AutoTokenizer.from_pretrained(model.hparams.encoder_model)

    return _model_cache[key], _tokenizer_cache[key]


def create_table_if_not_exists(conn_params: dict):
    """Create the article_embeddings table if it doesn't exist."""
    print(f"[DB] Connecting to {conn_params['host']}:{conn_params['port']} ...")
    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS article_embeddings (
            article_id      TEXT PRIMARY KEY,
            embedding       FLOAT4[],
            emotion_scores  FLOAT4[]
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("[DB] Table 'article_embeddings' ready.")


def upsert_rows(conn_params: dict, rows: list):
    """Bulk-upsert (article_id, embedding, emotion_scores) rows."""
    if not rows:
        return
    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()
    psycopg2.extras.execute_values(
        cur,
        """
        INSERT INTO article_embeddings (article_id, embedding, emotion_scores)
        VALUES %s
        ON CONFLICT (article_id) DO UPDATE
            SET embedding      = EXCLUDED.embedding,
                emotion_scores = EXCLUDED.emotion_scores
        """,
        rows,
        template="(%s, %s::float4[], %s::float4[])"
    )
    conn.commit()
    cur.close()
    conn.close()


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
        Process a Spark partition: generate embeddings and store them in PostgreSQL.

        Args:
            partition_iter (Iterator): Rows in the Spark partition.
            checkpoint_path (str): Model checkpoint path.
            model_name (str): Tokenizer model name.
            num_classes (int): Main output size.
            extra_dropout (float): Model dropout.
            num_groups (int): Number of emotion classes.
            max_length (int): Max token length.
            batch_size (int): Batch size for inference.
            conn_params (dict): PostgreSQL connection parameters.
            id_col (str): Article ID column.
            text_col (str): Article text column.

        Returns:
            Iterator: Empty iterator (required by Spark).

        Methodology:
            1. Load (or reuse) model and tokenizer.
             2. Accumulate rows into batches.
             3. Tokenize with overflow to handle long texts.
             4. Run inference on all chunks.
             5. Group chunk outputs back to original articles.
             6. Aggregate embeddings (mean) and emotions (max).
        Side Effects:
            - Runs model inference (PyTorch).
            - Writes embeddings and emotion scores to PostgreSQL.
    """

    print(f"[Worker] Loading model & tokenizer from {checkpoint_path} ...")
    model, tokenizer = _get_model_and_tokenizer(
        checkpoint_path, model_name, num_classes, extra_dropout, num_groups
    )
    print("[Worker] Model ready. Starting partition processing...")

    buffer = []
    total_processed = 0
    batch_num = 0
    partition_start = time.time()

    def flush(batch_rows):
        nonlocal total_processed, batch_num
        batch_num += 1
        t0 = time.time()

        ids = [r[id_col] for r in batch_rows]
        texts = [_clean_text(r[text_col]) for r in batch_rows]

        tokenized = tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=max_length,
            stride=50,
            return_overflowing_tokens=True,
            return_tensors="pt",
            add_special_tokens=True,
        )

        mapping = tokenized.pop("overflow_to_sample_mapping").numpy()
        num_chunks = len(mapping)

        with torch.no_grad():
            outputs = model.model(
                input_ids=tokenized['input_ids'],
                attention_mask=tokenized['attention_mask'],
                output_hidden_states=True
            )

            last_hidden = outputs.hidden_states[-1]  # [chunks, seq_len, 768]
            mask = tokenized['attention_mask'].unsqueeze(-1)  # [chunks, seq_len, 1]
            sum_hidden = (last_hidden * mask).sum(dim=1)  # [chunks, 768]
            count = mask.sum(dim=1).clamp(min=1e-9)  # [chunks, 1]
            cls_chunks = (sum_hidden / count).cpu().numpy()  # [chunks, 768]

            logits_aux = outputs.logits
            if logits_aux is not None:
                emotion_chunks = torch.sigmoid(logits_aux).cpu().numpy()
            else:
                emotion_chunks = np.zeros((len(mapping), num_groups), dtype=np.float32)

        grouped_cls = defaultdict(list)
        grouped_emo = defaultdict(list)

        for chunk_idx, original_article_idx in enumerate(mapping):
            grouped_cls[original_article_idx].append(cls_chunks[chunk_idx])
            grouped_emo[original_article_idx].append(emotion_chunks[chunk_idx])

        db_rows = []
        for i, article_id in enumerate(ids):
            if i not in grouped_cls:
                final_cls = np.zeros(768, dtype=np.float32)
                final_emo = np.zeros(num_groups, dtype=np.float32)
            else:
                final_cls = np.mean(grouped_cls[i], axis=0)
                final_emo = np.max(grouped_emo[i], axis=0)
            db_rows.append((article_id, final_cls.tolist(), final_emo.tolist()))

        upsert_rows(conn_params, db_rows)

        elapsed = time.time() - t0
        total_processed += len(db_rows)
        throughput = len(db_rows) / elapsed if elapsed > 0 else float('inf')

        print(
            f"    [Worker] Batch {batch_num} — "
            f"{len(db_rows)} articles | "
            f"{num_chunks} chunks | "
            f"{elapsed:.1f}s | "
            f"{throughput:.1f} art/s | "
            f"partition total: {total_processed}"
        )

    for row in partition_iter:
        r_dict = row.asDict() if hasattr(row, "asDict") else row
        buffer.append(r_dict)
        if len(buffer) >= batch_size:
            flush(buffer)
            buffer = []

    if buffer:
        flush(buffer)

    partition_elapsed = time.time() - partition_start
    print(
        f"[Worker] Partition complete — "
        f"{total_processed} articles in {batch_num} batches | "
        f"total time: {partition_elapsed:.1f}s"
    )

    return iter([])


def main():
    args = get_args()

    print(f"[Main] Using DB host: {args.db_host}:{args.db_port}")

    conn_params = {
        "host":     args.db_host,
        "port":     args.db_port,
        "dbname":   args.db_name,
        "user":     args.db_user,
        "password": args.db_password,
    }

    try:
        create_table_if_not_exists(conn_params)
    except Exception as e:
        print(f"[Main] CRITICAL: Could not connect to Postgres: {e}")
        print(f"[Main] TIP: Ensure Postgres is running and accepting connections on {args.db_host}:{args.db_port}")
        return

    spark = (
        SparkSession.builder
        .appName("EmotionModel-Inference")
        .config("spark.sql.shuffle.partitions", str(args.spark_partitions))
        .config("spark.driver.memory", args.driver_memory)
        .config("spark.executor.memory", args.executor_memory)
        .config("spark.python.worker.faulthandler.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    df = (
        spark.read
        .option("header", "true")
        .option("multiLine", "true")
        .option("escape", '"')
        .csv(args.input_csv)
        .select(col(args.id_column), col(args.text_column))
        .dropna(subset=[args.id_column, args.text_column])
        .repartition(args.spark_partitions)
    )

    total_articles = df.count()
    num_partitions = df.rdd.getNumPartitions()
    print(
        f"[Main] Loaded {total_articles} articles across {num_partitions} partitions "
        f"(~{total_articles // max(num_partitions, 1)} articles/partition). "
        f"Starting distributed embedding job..."
    )

    checkpoint_path_bc = spark.sparkContext.broadcast(args.model_checkpoint)
    conn_params_bc     = spark.sparkContext.broadcast(conn_params)

    job_start = time.time()

    df.rdd.mapPartitions(
        lambda partition: embed_partition(
            partition,
            checkpoint_path = checkpoint_path_bc.value,
            model_name      = args.model_name,
            num_classes     = args.num_classes,
            extra_dropout   = args.extra_dropout,
            num_groups      = args.num_groups,
            max_length      = args.max_length,
            batch_size      = args.batch_size,
            conn_params     = conn_params_bc.value,
            id_col          = args.id_column,
            text_col        = args.text_column,
        )
    ).count()

    job_elapsed = time.time() - job_start
    print(
        f"[Main] Pipeline complete — {total_articles} articles embedded in {job_elapsed:.1f}s "
        f"({total_articles / job_elapsed:.1f} art/s overall). All embeddings stored in PostgreSQL."
    )
    spark.stop()


if __name__ == "__main__":
    main()