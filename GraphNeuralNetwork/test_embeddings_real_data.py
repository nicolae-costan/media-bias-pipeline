import os
import sys
import psycopg2
import pandas as pd

# Add parent directory to sys.path so we can import from GraphNeuralNetwork and EmotionModels
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
emotion_dir = os.path.join(parent_dir, 'EmotionModels')
sys.path.append(parent_dir)
sys.path.append(emotion_dir)

from GraphNeuralNetwork.article_embeddings import embed_partition, create_table_if_not_exists
from dotenv import load_dotenv

def test_real_data_embeddings():
    print("1. Loading 5 real articles from merged_clean_data.csv...")
    csv_path = "../data/merged_clean_data.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist. Run data_prelucration.ipynb first.")
        return
        
    df = pd.read_csv(csv_path).head(5)
    
    # Convert DataFrame rows to a list of dictionaries (which mimics a Spark partition)
    dummy_partition = df.to_dict('records')
    
    print(f"   Loaded {len(dummy_partition)} articles.")
    for i, row in enumerate(dummy_partition):
        print(f"   - Article {i+1} ID: {row['article_id']}")

    print("\n2. Setting up database connection parameters...")
    load_dotenv()
    conn_params = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", 5433)),
        "dbname": os.getenv("DB_NAME", "media_bias"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "")
    }

    create_table_if_not_exists(conn_params)

    print("3. Cleaning up database to prepare for a fresh test...")
    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()
    
    # Delete these 5 articles if they already exist from a previous run
    article_ids = [row["article_id"] for row in dummy_partition]
    cur.execute("DELETE FROM article_embeddings WHERE article_id = ANY(%s);", (article_ids,))
    conn.commit()

    checkpoint_path = os.getenv("MODEL_CHECKPOINT")
    print(f"\n4. Running PySpark embed_partition logic with checkpoint:\n   {checkpoint_path}")
    
    try:
        # Run the partition logic
        result_iter = embed_partition(
            partition_iter=dummy_partition,
            checkpoint_path=checkpoint_path,
            model_name="SamLowe/roberta-base-go_emotions",
            num_classes=1,
            extra_dropout=0.0,
            num_groups=13,
            max_length=128,
            batch_size=32,
            conn_params=conn_params,
            id_col="article_id",
            text_col="body"
        )
        # Consume the generator to actually execute the code
        list(result_iter)
        
    except Exception as e:
        print(f"FAILED during embed_partition: {e}")
        return

    print("\n5. Verifying database insertions...")
    cur.execute("SELECT article_id, array_length(embedding, 1), array_length(emotion_scores, 1) FROM article_embeddings WHERE article_id = ANY(%s);", (article_ids,))
    rows = cur.fetchall()
    
    cur.close()
    conn.close()

    if len(rows) == 5:
        print("\n--- RESULTS ---")
        for row_idx, row in enumerate(rows):
            fetched_id, emb_len, emo_len = row
            print(f"[{row_idx+1}] ID: {fetched_id[:10]}... | Embedding Size: {emb_len} | Emotion Size: {emo_len}")
            assert emb_len == 768, f"Embedding length mismatch for {fetched_id}"
            assert emo_len == 13, f"Emotion scores length mismatch for {fetched_id}"
            
        print("\nSUCCESS: All 5 real articles were perfectly processed and inserted into PostgreSQL!")
    else:
        print(f"\nFAILURE: Expected 5 records in the database, but found {len(rows)}.")

if __name__ == "__main__":
    test_real_data_embeddings()
