# Media Bias Analysis Pipeline

A multi-task learning pipeline for detecting media bias and emotions in news articles and social media text.

## 🚀 Quick Start (Database)

To get the PostgreSQL + pgvector database running on Windows:

1.  **Start Docker Desktop**.
2.  Run the automated startup script:
    ```powershell
    .\start_db.ps1
    ```
3.  **Initialize the Schema**:
    ```powershell
    python GraphNeuralNetwork/setup_db.py
    ```

## 📂 Project Structure

- **`SentimentClassification/`**: Bias detection in Reddit comments (Us vs Them scale).
- **`EmotionModels/`**: Emotion classification in news articles (13 emotions).
- **`GraphNeuralNetwork/`**: GNN-based bias classification using semantic similarity and metadata.
- **`data/`**: Datasets including BABE and UsVsThem.

## 📊 Interpretability & Analysis

To analyze what the models are actually learning (Token Ablation Analysis):

1.  **Emotion Analysis**:
    ```powershell
    python interpret_models.py --model_type emotion --checkpoint "path/to/emotion.ckpt"
    ```
2.  **Bias Analysis**:
    ```powershell
    python interpret_models.py --model_type bias --checkpoint "path/to/bias.ckpt"
    ```

---

## 🛠️ Detailed Documentation

- [**Architecture Overview**](ARCHITECTURE_OVERVIEW.md): How the pieces fit together.
- [**Database Migration Report**](DATABASE_MIGRATION_REPORT.md): Details on the PostgreSQL + pgvector setup.
- [**Codebase Deep Dive**](codebase_explanation.md): A step-by-step beginner's guide to the logic.