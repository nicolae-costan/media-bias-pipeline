import os
import sys
import argparse
import numpy as np
import psycopg2
import psycopg2.extras
from pyspark.sql import SparkSession
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()


def create_mock_table(conn_params):
    print(f"Connecting to {conn_params['host']}:{conn_params['port']}...")
    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()
    cur.execute("""
                CREATE TABLE IF NOT EXISTS mock_article_embeddings
                (
                    article_id
                    TEXT
                    PRIMARY
                    KEY,
                    embedding
                    FLOAT4[],
                    emotion_scores
                    FLOAT4
                []
                );
                """)
    conn.commit()
    cur.close()
    conn.close()
    print("Table 'mock_article_embeddings' ready.")


def upsert_mock_rows(conn_params, rows):
    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()
    psycopg2.extras.execute_values(
        cur,
        "INSERT INTO mock_article_embeddings VALUES %s ON CONFLICT DO NOTHING",
        rows,
        template="(%s, %s::float4[], %s::float4[])"
    )
    conn.commit()
    cur.close()
    conn.close()


def main():
    parser = argparse.ArgumentParser()
    # Defaults to the DB_HOST from .env, or 'localhost' if not found.
    parser.add_argument("--db_host", type=str, default=os.getenv("DB_HOST", "localhost"))
    args = parser.parse_args()

    print(f"Using DB Host: {args.db_host}")

    conn_params = {
        "host": args.db_host,
        "port": int(os.getenv("DB_PORT", 5433)),  # Change to 5432 if you reverted to default Postgres port
        "dbname": os.getenv("DB_NAME", "media_bias"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "")
    }

    # 1. Test Connection and Create Table
    try:
        create_mock_table(conn_params)
    except Exception as e:
        print(f"CRITICAL: Could not connect to Postgres: {e}")
        print(
            f"\nTIP: Ensure Postgres is running on Linux and accepting connections on {args.db_host}:{conn_params['port']}.")
        return

    # 2. Start Mock Spark Session
    spark = SparkSession.builder.appName("Mock-Postgres-Write").getOrCreate()

    # Create 10 dummy articles
    data = [(f"id_{i}", "Some text body") for i in range(10)]
    df = spark.createDataFrame(data, ["article_id", "body"])

    def process_partition(iter):
        db_rows = []
        for row in iter:
            dummy_emb = np.random.rand(768).tolist()
            dummy_emo = np.random.rand(13).tolist()
            db_rows.append((row.article_id, dummy_emb, dummy_emo))

        upsert_mock_rows(conn_params, db_rows)
        return iter

    print("Running Spark Job...")
    df.rdd.mapPartitions(process_partition).count()
    print("Success! Dummy data written to native Linux Postgres.")
    spark.stop()


if __name__ == "__main__":
    main()