Article Embedding Pipeline

A distributed text embedding pipeline built with PySpark and a Transformer model.
It processes articles from a CSV file, generates embeddings and emotion scores, and stores them in PostgreSQL.


PostgreSQL table: article_embeddings

Column	Type	Description
article_id	TEXT	Primary key
embedding	FLOAT4[]	Article embedding vector
emotion_scores	FLOAT4[]	Emotion probability vector

How to Run
1. Configure the run script

Edit the provided shell script:

INPUT_CSV="articles.csv"
MODEL_CKPT="path/to/your_model.ckpt"

DB_NAME="media_bias"
DB_USER="postgres"
DB_PASS="your_password"
DB_HOST="localhost"
DB_PORT=5432

Adjust performance settings if needed:

BATCH_SIZE=16
NUM_PARTITIONS=4
DRIVER_RAM="2g"
EXEC_RAM="3g"
2. Execute the pipeline

Run the script:

bash run.sh

Or run directly with Spark:

spark-submit embed_articles.py \
  --input_csv articles.csv \
  --model_checkpoint path/to/model.ckpt \
  --db_name media_bias \
  --db_user postgres \
  --db_password your_password
Notes
The model and tokenizer are cached per Spark worker to reduce overhead
Long texts are split into overlapping chunks before inference
Embeddings are averaged; emotion scores use max pooling
The pipeline is safe to re-run due to upsert logic