#!/bin/bash



# Data Paths
INPUT_CSV="articles.csv"
MODEL_CKPT="path/to/your_model.ckpt"

# Database Credentials
DB_NAME="media_bias"
DB_USER="postgres"
DB_PASS="your_password"
DB_HOST="localhost"
DB_PORT=5432

# Model Architecture Settings
MODEL_TYPE="bert-base-uncased"
EMOTION_GROUPS=13
MAIN_CLASSES=1
DROPOUT=0.0
MAX_TEXT_LEN=512

# Hardware & Scaling Settings (Adjust for your RAM)
DRIVER_RAM="2g"
EXEC_RAM="3g"
BATCH_SIZE=16
NUM_PARTITIONS=4

# ==============================================================================
# EXECUTION COMMAND
# ==============================================================================

spark-submit embed_articles.py \
    --input_csv "$INPUT_CSV" \
    --model_checkpoint "$MODEL_CKPT" \
    --model_name "$MODEL_TYPE" \
    --id_column "article_id" \
    --text_column "body" \
    --num_groups $EMOTION_GROUPS \
    --num_classes $MAIN_CLASSES \
    --extra_dropout $DROPOUT \
    --max_length $MAX_TEXT_LEN \
    --db_host "$DB_HOST" \
    --db_port $DB_PORT \
    --db_name "$DB_NAME" \
    --db_user "$DB_USER" \
    --db_password "$DB_PASS" \
    --driver_memory "$DRIVER_RAM" \
    --executor_memory "$EXEC_RAM" \
    --batch_size $BATCH_SIZE \
    --spark_partitions $NUM_PARTITIONS