"""
setup_db.py
-----------
Run this ONCE on a fresh environment to create the database schema.
It installs the pgvector extension and creates the two tables.

Usage:
    python GraphNeuralNetwork/setup_db.py
"""

import os
import sys
import psycopg2
from dotenv import load_dotenv

# Load .env from the same directory as this script
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)


def get_conn_params():
    return {
        "host":     os.getenv("DB_HOST", "localhost"),
        "port":     int(os.getenv("DB_PORT", 5433)),
        "dbname":   os.getenv("DB_NAME", "media_bias"),
        "user":     os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", ""),
    }


def setup(conn_params: dict):
    print(f"Connecting to PostgreSQL at {conn_params['host']}:{conn_params['port']} ...")
    try:
        conn = psycopg2.connect(**conn_params)
    except psycopg2.OperationalError as e:
        print(f"\nERROR: Could not connect to the database.\n{e}")
        print("\nMake sure the Docker container is running:")
        print("  docker start media-bias-postgres")
        sys.exit(1)

    conn.autocommit = True
    cur = conn.cursor()

    # 1. Install pgvector extension
    print("Installing pgvector extension...")
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    print("  [OK] vector extension ready")

    # 2. Articles table — stores article metadata
    print("Creating 'articles' table...")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            article_id   TEXT PRIMARY KEY,
            body         TEXT,
            outlet       TEXT,
            topic        TEXT,
            type         TEXT,
            label_bias   TEXT,
            news_link    TEXT
        );
    """)
    print("  [OK] articles table ready")

    # 3. Article embeddings table — stores ML outputs, references articles
    print("Creating 'article_embeddings' table...")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS article_embeddings (
            article_id     TEXT PRIMARY KEY REFERENCES articles(article_id) ON DELETE CASCADE,
            embedding      VECTOR(768),
            emotion_scores FLOAT4[]
        );
    """)
    print("  [OK] article_embeddings table ready")

    # 4. Useful index for fast similarity search
    print("Creating HNSW index for fast vector search...")
    cur.execute("""
        CREATE INDEX IF NOT EXISTS article_embeddings_embedding_idx
        ON article_embeddings
        USING hnsw (embedding vector_cosine_ops);
    """)
    print("  [OK] HNSW index ready")


    cur.close()
    conn.close()
    print("\nDatabase setup complete.")


if __name__ == "__main__":
    setup(get_conn_params())
