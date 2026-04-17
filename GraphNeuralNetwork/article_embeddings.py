import argparse
import re

import numpy as np
import torch
import psycopg2
from matplotlib import collections
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import (
    StructType, StructField, StringType,
    ArrayType, FloatType
)
from collections import defaultdict
import pandas as pd
from transformers import AutoTokenizer



def get_args():
    """
       Parse command-line arguments for the embedding pipeline.

       Configures input data paths, model parameters, Spark settings,
       and PostgreSQL connection details required for running the job.

       Returns:
           argparse.Namespace: Parsed arguments containing all runtime configuration.

       Side Effects:
           - Reads CLI arguments from sys.argv.

       Notes:
           - Enforces required arguments such as input CSV, model checkpoint,
             and database credentials.
       """
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
    parser.add_argument("--driver_memory", type=str, default="2g", help="RAM for driver")
    parser.add_argument("--executor_memory", type=str, default="3g", help="RAM for executor")
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
    """
      Load and cache the model and tokenizer for inference.

      Initializes a RedditTransformer model from a checkpoint and pairs it with
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

        return _model_cache[key], _tokenizer_cache[key]

def create_table_if_not_exists(conn_params: dict):
    """Create the article_embeddings table if it doesn't exist."""
    conn = psycopg2.connect(**conn_params)
    cur  = conn.cursor()
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


def upsert_rows(conn_params: dict, rows: list):
    """
    Bulk-upsert a list of (article_id, embedding, emotion_scores) tuples.
    Uses ON CONFLICT DO UPDATE so re-runs are safe.
    """
    if not rows:
        return
    conn = psycopg2.connect(**conn_params)
    cur  = conn.cursor()
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
    model, tokenizer = _get_model_and_tokenizer(
        checkpoint_path, model_name, num_classes, extra_dropout, num_groups
    )

    buffer = []

    def flush(batch_rows):

        ids = [r[id_col] for r in batch_rows]
        texts = [_clean_text(r[text_col]) for r in batch_rows]

        # for long sentences split them with overlap of 50
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

        # using the dictionary of tokenized get a list [0,0,1] where 0 is article 1 that has 2 chunks
        mapping = tokenized.pop("overflow_to_sample_mapping").numpy()



        with torch.no_grad():
            # Run all chunks through the model simultaneously
            logits_main, logits_aux, hidden_states = model(tokenized)

            # Extract CLS embeddings for all chunks [Num_Chunks, 768]
            # we use the us vs them main task layer 12 embeddings CLS token because it can detect bias and hostility better than just sentiments and by doing this we have a more robust model

            if isinstance(hidden_states, tuple):
                cls_chunks = hidden_states[0][:, 0, :].cpu().numpy()
            else:
                cls_chunks = hidden_states[:, 0, :].cpu().numpy()

            # Extract Emotion vectors for all chunks [Num_Chunks, 13]
            if logits_aux is not None:
                # run sigmoid like to get a score
                emotion_chunks = torch.sigmoid(logits_aux).cpu().numpy()
            else:
                emotion_chunks = np.zeros((len(mapping), num_groups), dtype=np.float32)

        grouped_cls = defaultdict(list)
        grouped_emo = defaultdict(list)

        # aggregate groups
        for chunk_idx, original_article_idx in enumerate(mapping):
            grouped_cls[original_article_idx].append(cls_chunks[chunk_idx])
            grouped_emo[original_article_idx].append(emotion_chunks[chunk_idx])


        db_rows = []
        for i,article_id in enumerate(ids):

            if i not in grouped_cls:
                final_cls = np.zeros(768, dtype=np.float32)
                final_emo = np.zeros(num_groups, dtype=np.float32)
            else:
                # Average (Mean) the CLS embeddings
                final_cls = np.mean(grouped_cls[i], axis=0)

                # Max the Emotion scores (Find the most extreme spike of emotion)
                final_emo = np.max(grouped_emo[i], axis=0)
            db_rows.append((
                article_id,
                final_cls.tolist(),
                final_emo.tolist()
            ))

        # save data to database
        upsert_rows(conn_params,db_rows)

        for row in partition_iter:
            buffer.append(row.asDict())
            if len(buffer) >= batch_size:
                flush(buffer)
                buffer = []
            if buffer:
                flush(buffer)

        return iter([])

def main():
    args = get_args()

    conn_params = {
        "host":     args.db_host,
        "port":     args.db_port,
        "dbname":   args.db_name,
        "user":     args.db_user,
        "password": args.db_password,
    }

    create_table_if_not_exists(conn_params)

    spark = (
        SparkSession.builder
        .appName("RedditTransformer-Inference")
        .config("spark.sql.shuffle.partitions", str(args.spark_partitions))
        .config("spark.driver.memory", args.driver_memory)
        .config("spark.executor.memory", args.executor_memory)
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

    checkpoint_path_bc = spark.sparkContext.broadcast(args.model_checkpoint)
    conn_params_bc     = spark.sparkContext.broadcast(conn_params)

    df.rdd.mapPartitions(
        lambda partition: embed_partition(
            partition,
            checkpoint_path = checkpoint_path_bc.value,  # Get from broadcast
            model_name      = args.model_name,
            num_classes     = args.num_classes,
            extra_dropout   = args.extra_dropout,
            num_groups      = args.num_groups,
            max_length      = args.max_length,
            batch_size      = args.batch_size,
            conn_params     = conn_params_bc.value,      # Get from broadcast
            id_col          = args.id_column,
            text_col        = args.text_column,
        )
    ).count()  # This 'Action' tells Spark to start the engine

    print("[embed_articles] Success: Data written to PostgreSQL.")
    spark.stop()

if __name__ == "__main__":
    main()
