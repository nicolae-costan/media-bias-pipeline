# Pipeline — End-to-End Commands

This document is the **operational runbook**: every command needed to take the project from a fresh clone to a trained pipeline with interpretability outputs. Commands are listed in the order they should be executed.

> **Assumptions** — Linux/macOS shell or Git Bash on Windows. Replace `python` with `python3` if your environment requires it. All paths are relative to the repository root unless noted.

---

## 0. Prerequisites

```bash
# Required system tools
docker --version          # Docker 20+ with Compose v2
python --version          # Python 3.10+ recommended
```

GPU is optional but strongly recommended for the Transformer training stages.

---

## 1. Environment Setup

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate                # Windows (cmd):   venv\Scripts\activate
                                        # Windows (PS):    venv\Scripts\Activate.ps1

# Install all dependencies (root requirements.txt is consolidated)
pip install --upgrade pip
pip install -r requirements.txt
```

Edit `.env` if you change the database password, port, or dataset paths. The committed defaults match `docker-compose.yml`.

---

## 2. Database Layer (PostgreSQL + pgvector)

```bash
# Bring up the Postgres container (port 5433 on the host)
chmod +x ./start_db.sh
./start_db.sh

# Create tables, vector columns, and HNSW indexes
python GraphNeuralNetwork/setup_db.py
```

Sanity check the container:

```bash
docker ps | grep media-bias-postgres
docker exec -it media-bias-postgres psql -U postgres -d media_bias -c "\dt"
```

To tear it down and start fresh:

```bash
docker compose down -v        # WARNING: -v wipes the volume / all data
```

---

## 3. Data Preparation

### 3.1 Consensus merge (PySpark)

```bash
# Combines raw_labels_SG1.csv + raw_labels_SG2.csv → consensus_labels_sg1_sg2.csv
python merge_sg.py
```

### 3.2 Bias-transformer splits

```bash
# Produces pretrain_*.csv and finetune_*.csv in data/bias_transformer/
python BiasTransformer/prepare_data.py
```

---

## 4. Stage A — Emotion Classifier

```bash
cd EmotionModels

# Train the 7-class emotion model (test-tube HyperOptArgumentParser flags)
python train.py \
  --gpus 1 \
  --max_epochs 10 \
  --batch_size 16 \
  --learning_rate 2e-5

# Evaluate the best checkpoint on the held-out split
python test.py --checkpoint tb_logs/emotion_classification/version_0/checkpoints/<best>.ckpt

# Fit per-class optimal decision thresholds → thresholds.json
python optimize_thresholds.py --checkpoint tb_logs/emotion_classification/version_0/checkpoints/<best>.ckpt

cd ..
```

> Replace `<best>` with the actual checkpoint filename from the `tb_logs/.../checkpoints/` directory.

---

## 5. Stage B — Article Embeddings → pgvector

```bash
cd GraphNeuralNetwork

# Embed every article with the fine-tuned Transformer and UPSERT into pgvector
python article_embeddings.py \
  --checkpoint ../EmotionModels/tb_logs/emotion_classification/version_0/checkpoints/<best>.ckpt

# Smoke-test the vector store
python test_embeddings_real_data.py
python ../test_real_search.py

cd ..
```

---

## 6. Stage C — Graph Construction

```bash
cd GraphNeuralNetwork

# Build the similarity graph; outputs graph.pt at the repo root
python build_graph.py \
  --top_k 15 \
  --sim_threshold 0.5 \
  --split_mode random_stratified \
  --train_frac 0.70 \
  --val_frac 0.15 \
  --output ../graph.pt

cd ..
```

All flags here can also be set via `.env` (see `TOP_K`, `SIM_THRESHOLD`, `SPLIT_MODE`, etc.).

---

## 7. Stage D — Graph Attention Network

```bash
cd GraphNeuralNetwork

# Train the GAT on the constructed graph
python train.py \
  --graph_path ../graph.pt \
  --max_epochs 100 \
  --lr_gat 1e-3 \
  --accelerator gpu --devices 1

# Test the best GAT checkpoint
python test.py --graph_path ../graph.pt --checkpoint checkpoints/<best>.ckpt

# Random Forest baseline (non-graph reference)
python random_forest_baseline.py --graph_path ../graph.pt

cd ..
```

---

## 8. Stage E — Final Bias Transformer

```bash
cd BiasTransformer

# Generate GNN pseudo-labels for unlabeled articles
python graph_pseudo_labels.py \
  --graph_path ../graph.pt \
  --gnn_checkpoint ../GraphNeuralNetwork/checkpoints/<best>.ckpt

# Two-stage training: pretrain on pseudo-labels, then fine-tune on consensus gold
python train.py \
  --max_epochs 5 \
  --gpus 1 \
  --data_dir ../data/bias_transformer

# Skip the pretrain stage if you already have a checkpoint
python train.py \
  --skip_pretrain \
  --pretrained_checkpoint tb_logs/bias_transformer_pretrain/version_0/checkpoints/<best>.ckpt

# Evaluate
python test.py --checkpoint tb_logs/bias_transformer/version_0/checkpoints/<best>.ckpt

cd ..
```

---

## 9. Stage F — Interpretability

```bash
# Token-level ablation for the emotion classifier
python interpret_models.py \
  --model_type emotion \
  --checkpoint EmotionModels/tb_logs/emotion_classification/version_0/checkpoints/<best>.ckpt

# Token-level ablation for the bias classifier
python interpret_models.py \
  --model_type bias \
  --checkpoint BiasTransformer/tb_logs/bias_transformer/version_0/checkpoints/<best>.ckpt
```

---

## 10. Monitoring

```bash
# TensorBoard for any stage (point at the parent tb_logs directory)
tensorboard --logdir tb_logs
tensorboard --logdir EmotionModels/tb_logs
tensorboard --logdir GraphNeuralNetwork/tb_logs
tensorboard --logdir BiasTransformer/tb_logs
```

---

## 11. Spark Cluster Smoke Test (optional)

```bash
cd GraphNeuralNetwork
./run_spark.sh
python spark_database_test.py
cd ..
```

---

## End-to-End TL;DR

The minimum command sequence to go from a clean repo to a trained pipeline:

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
./start_db.sh
python GraphNeuralNetwork/setup_db.py
python merge_sg.py
python BiasTransformer/prepare_data.py
python EmotionModels/train.py --gpus 1 --max_epochs 10
python EmotionModels/optimize_thresholds.py --checkpoint <emotion.ckpt>
python GraphNeuralNetwork/article_embeddings.py --checkpoint <emotion.ckpt>
python GraphNeuralNetwork/build_graph.py --output graph.pt
python GraphNeuralNetwork/train.py --graph_path graph.pt
python BiasTransformer/graph_pseudo_labels.py --graph_path graph.pt --gnn_checkpoint <gnn.ckpt>
python BiasTransformer/train.py
python interpret_models.py --model_type bias --checkpoint <bias.ckpt>
```
