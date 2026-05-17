# Media Bias and Semantic Analysis Pipeline

A state-of-the-art multi-stage machine learning pipeline combining **Deep Transformers (BERT/RoBERTa)** and **Graph Attention Networks (GAT)** to analyze, classify, and interpret media bias and emotional signaling in journalism and social media text.

---

## 🚀 Quick Start (Database Layer Setup)

The database layer utilizes a containerized **PostgreSQL** deployment configured with the **`pgvector`** extension to handle dense vector storage and high-speed similarity indexing.

1.  **Initialize the database container:**
    ```bash
    chmod +x ./start_db.sh
    ./start_db.sh
    ```
2.  **Initialize the database schema (Tables, Keys, and HNSW indexes):**
    ```bash
    python GraphNeuralNetwork/setup_db.py
    ```

---

## 📂 Project Architecture Overview

*   **`utils/`**: Distributed data preprocessing routines using the **PySpark** engine to resolve consensus annotations (SG1/SG2) via majority voting.
*   **`GraphNeuralNetwork/`**: Semi-supervised node classification using Graph Attention Networks (GAT) trained on semantic similarity and emotional distributions of news articles.
*   **`EmotionModels/`**: Fine-tuned multi-label transformer models (using Asymmetric Loss and optimal classification thresholds) to detect emotional signals in media.
*   **`SentimentClassification/`**: Baseline multi-task continuous regression models (BERT/RoBERTa) trained on Reddit "Us vs Them" data to represent continuous ideological polarity.

---

## 📊 Interpretability & Influence Analysis

To analyze the semantic decision boundaries of fine-tuned models, run the token-level ablation (masking) interpreter:

1.  **Extract Word Influences from the Emotion Classifier:**
    ```bash
    python interpret_models.py --model_type emotion --checkpoint "path/to/emotion.ckpt"
    ```
2.  **Extract Word Influences from the Baseline Bias Classifier:**
    ```bash
    python interpret_models.py --model_type bias --checkpoint "path/to/bias.ckpt"
    ```

---

## 🛠️ Detailed Documentation

*   [**Technical Architecture Overview**](ARCHITECTURE_OVERVIEW.md): Comprehensive system maps and data structures.
*   [**Database Architecture Migration**](DATABASE_MIGRATION_REPORT.md): Relational pgvector design and schema breakdown.
*   [**Codebase Deep Dive**](codebase_explanation.md): Step-by-step logic walks through the codebase.