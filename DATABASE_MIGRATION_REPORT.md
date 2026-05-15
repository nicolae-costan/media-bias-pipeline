# Database Migration: ChromaDB → PostgreSQL + pgvector

This document explains every change made to the database layer of the pipeline, why each change was made, and how the pieces fit together.

---

## The Problem: Why We Changed Anything

On the `feature/PlotsAndVisualization` branch, the pipeline was **broken end-to-end** in a subtle way:

- `article_embeddings.py` had been rewritten to write article embeddings into **ChromaDB**, a standalone vector database that stores its data in a local folder (`./vector_db/`).
- But `build_graph.py` — the script that reads those embeddings to build the similarity graph — **was never updated**. It still tries to connect to **PostgreSQL** and run a SQL query.

This means the two scripts were talking to completely different databases. You could run `article_embeddings.py` successfully and fill ChromaDB with data, but then `build_graph.py` would fail immediately because PostgreSQL has nothing in it. The graph could never be built.

Beyond the broken pipeline, ChromaDB was the wrong tool for this job for a deeper reason: it is a pure vector store. It is very good at storing embeddings and finding similar ones, but it cannot handle relational queries. You cannot ask ChromaDB "show me all articles from Fox News published this month" in any reasonable way. PostgreSQL can do that trivially with a two-line SQL query.

---

## Step 1: Choosing the Right Database Setup

PostgreSQL is already part of this project — `build_graph.py` was written for it. The missing ingredient was **pgvector**, an open-source extension for PostgreSQL that adds a new column type called `VECTOR(n)`. Once installed, a `VECTOR(768)` column behaves exactly like any other SQL column, except it can also be searched by cosine similarity using the `<=>` operator.

This means we get the best of both worlds: the relational power of PostgreSQL (joins, filters, dates, outlets) and the vector search capability that ChromaDB was being used for.

Since PostgreSQL was not running on your Windows machine (it was originally set up inside Linux/WSL, which was stopped), we used **Docker** to spin up a container running the official `pgvector/pgvector:pg16` image. This is a standard PostgreSQL 16 image that comes with pgvector pre-installed. The container is called `media-bias-postgres` and it listens on port **5433** (we use 5433 instead of the default 5432 to avoid clashing with anything else that might use that port).

---

## Step 2: Designing the Schema — Two Tables Instead of One

The original PostgreSQL schema (from the old `main` branch) had a single table called `article_embeddings` with three columns: `article_id`, `embedding`, and `emotion_scores`. That's it. No outlet, no date, no topic.

We redesigned this into **two separate tables**:

### `articles` table

This stores everything we know about an article as a piece of text — who published it, what it is about, whether it is biased. This data comes from the CSV files and does not change when you retrain the model.

```
article_id   → the unique hash ID for the article (primary key)
body         → the full article text
outlet       → the news outlet, e.g. "fox-news", "nytimes", "huffpost"
topic        → the subject, e.g. "immigration", "abortion", "elections-2020"
type         → political leaning: "left", "right", or "center"
label_bias   → "Biased" or "Non-biased"
news_link    → URL of the original article
```

### `article_embeddings` table

This stores the outputs of the machine learning model — the embedding vector and the emotion scores. These CAN change if you retrain the EmotionModel, so they live separately from the article metadata.

```
article_id     → references articles.article_id (foreign key)
embedding      → VECTOR(768), the 768-number meaning fingerprint of the article
emotion_scores → FLOAT4[], the 13 emotion probability scores
```

**Why two tables?** Because they represent different things with different lifecycles. The outlet of a Fox News article will not change. But if you retrain the model tomorrow with a better architecture, all the embeddings change. With two tables, you can re-run the embedding step and update `article_embeddings` without touching `articles` at all. If they were in one table, every retrain would force you to re-populate all the metadata too.

There is also a **HNSW index** on the `embedding` column. HNSW (Hierarchical Navigable Small World) is the algorithm pgvector uses to find the nearest neighbours of a vector efficiently. Without it, finding the 10 most similar articles to a given one would require comparing against all 13,697 articles one by one. With the HNSW index, it takes milliseconds.

---

## Step 3: Where Does the Metadata Come From?

The main data file (`data/merged_clean_data.csv`) only has three columns: `article_id`, `body`, and `label_bias`. It does not have outlet or topic.

However, the BABE dataset (`data/final_labels_MBIC.csv`) does — it has `outlet`, `topic`, `type`, and `news_link` for every article it covers. BABE is a subset of the full 13,697 articles, covering about 1,700 of them.

The new `article_embeddings.py` does a **left join** between these two files on `article_id`. For the ~1,700 articles that appear in BABE, the outlet and topic columns get populated. For the remaining ~12,000 articles that only exist in the main CSV, those columns are left as `NULL`.

---

## Step 4: Files Created or Changed

### New file: `GraphNeuralNetwork/.env`

This is the real credentials file (as opposed to `.env_copy` which is just a template). It points to the Docker container:

- Host: `localhost`
- Port: `5433`
- Database name: `media_bias`
- User: `postgres`
- Password: `mediabias123`

This file is already listed in `.gitignore` so it will not be accidentally committed with your credentials.

### New file: `GraphNeuralNetwork/setup_db.py`

A small one-time script that connects to PostgreSQL and sets up the schema. You run it once on a fresh environment (new computer, fresh Docker container, etc.) before anything else. It does four things in order:

1. Installs the pgvector extension: `CREATE EXTENSION IF NOT EXISTS vector`
2. Creates the `articles` table
3. Creates the `article_embeddings` table with the `VECTOR(768)` column
4. Creates the HNSW index for fast similarity search

### Rewritten: `GraphNeuralNetwork/article_embeddings.py`

The entire file was rewritten. The structure of what it does is the same (read CSV → run model → store results), but everything about where it stores is different:

- Removed all ChromaDB imports and calls
- Added a PostgreSQL connection using `psycopg2`
- Added logic to load and join the BABE CSV for metadata
- The processing loop now calls `upsert_batch()` which writes to both the `articles` table and the `article_embeddings` table in one go
- The upsert logic for `articles` uses `ON CONFLICT DO NOTHING` — meaning if you run the script twice, it will not overwrite metadata you might have manually edited. For `article_embeddings` it uses `ON CONFLICT DO UPDATE` — meaning re-running will update the embedding and emotion scores to reflect any model changes.

### Small fix: `GraphNeuralNetwork/build_graph.py`

Only one thing changed here. The old `article_embeddings` table had the embedding stored as a PostgreSQL array (`FLOAT4[]`), which psycopg2 automatically converts to a Python list. The new table stores it as a pgvector `VECTOR(768)` type, which psycopg2 returns as a raw string that looks like `"[0.1,0.2,0.3,...]"`.

A small helper function called `_parse_vector()` was added that detects this string and converts it back to a list of floats. This is a one-line parsing step; everything else in `build_graph.py` is unchanged.

### Updated: `GraphNeuralNetwork/requirements.txt`

Added `pgvector` — the Python package that registers the vector type with psycopg2 so it handles the `VECTOR` column correctly.

### Updated: `EmotionModels/requirements.txt`

Removed `chromadb`. It was only used by `visualize_emotions.py`, which will be replaced when we tackle the visualisation improvements later.

---

## Step 5: How to Start the Database

The Docker container needs to be running before any scripts that touch the database. You only need to start it, not re-create it — the data persists inside the container.

```powershell
# Start the container (after a reboot or if it stopped)
docker start media-bias-postgres

# Check it is running
docker ps
```

If you ever need to connect to it directly (for example with a GUI tool such as TablePlus or DBeaver), the connection details are:

```
Host:     localhost
Port:     5433
Database: media_bias
User:     postgres
Password: mediabias123
```

---

## Setting Up on a New Machine (for teammates pulling this branch)

The Docker container does not get committed to Git — it only exists on the machine where it was created. When someone pulls this branch for the first time, they need to run the following commands once:

```powershell
# 1. Pull and start the pgvector container (only needed the very first time)
docker run -d `
  --name media-bias-postgres `
  -e POSTGRES_USER=postgres `
  -e POSTGRES_PASSWORD=mediabias123 `
  -e POSTGRES_DB=media_bias `
  -p 5433:5432 `
  pgvector/pgvector:pg16

# 2. Copy the env template and fill in credentials
# (The defaults already match the docker run command above)
copy GraphNeuralNetwork\.env_copy GraphNeuralNetwork\.env

# 3. Install the Python database packages
pip install psycopg2-binary pgvector

# 4. Create the schema (tables + indexes)
python GraphNeuralNetwork/setup_db.py
```

After that, `docker start media-bias-postgres` is all that is needed on subsequent runs (after a reboot, etc.). The data inside the container persists between starts.

---

## What the Pipeline Looks Like Now

```
Step 0 (once per environment):
  docker start media-bias-postgres
  python GraphNeuralNetwork/setup_db.py
    → creates tables and HNSW index

Step 1:
  python EmotionModels/train.py
    → trains the Emotion Model, saves a checkpoint

Step 2:
  python GraphNeuralNetwork/article_embeddings.py \
      --input_csv "./data/merged_clean_data.csv" \
      --babe_csv  "./data/final_labels_MBIC.csv" \
      --model_checkpoint "path/to/checkpoint.ckpt"
    → for each article:
        runs BERT → 768-dim embedding (mean pooling over last hidden state)
        runs classifier head → 13 emotion probability scores
    → writes to articles table (with outlet, topic, type from BABE where available)
    → writes to article_embeddings table (VECTOR(768) + FLOAT4[13])

Step 3:
  python GraphNeuralNetwork/build_graph.py
    → reads embeddings from article_embeddings table
    → parses VECTOR strings back to numpy arrays
    → builds k-NN similarity graph
    → saves graph.pt

Step 4:
  python GraphNeuralNetwork/train.py
    → loads graph.pt
    → trains the Graph Attention Network
```

The key difference from before: steps 2 and 3 now talk to the **same** database. The pipeline is no longer broken.
