# Media Bias Pipeline

> A full-stack AI pipeline that detects political and ideological bias in news articles — from raw text to cross-platform visualization.

---

## Project Vision

The media landscape is increasingly polarized. This project builds an end-to-end system that **automatically detects, quantifies, and visualizes media bias** in news articles. Rather than relying on hand-crafted rules, every component is powered by machine learning — from emotion detection to graph-based reasoning and final prediction.

The pipeline is composed of four major stages:

```
News Article
    |
    v
[1] Emotion & Sentiment Classification
    |       (BERT / RoBERTa fine-tuned on Reddit comments)
    v
[2] Graph Neural Network (Semi-Supervised Bias Propagation)
    |       (GNN trained on entity/article relationship graphs)
    v
[3] Media Bias Prediction Model
    |       (Final bias score aggregation and classification)
    v
[4] FastAPI Backend  ────────►  C# MAUI Frontend
        (REST API)               (News API integration + UI)
```

---

## Stage 1 — Emotion & Sentiment Classification

**Location:** `SentimentClassification/`, `EmotionModels/`

The first stage of the pipeline teaches the system to understand the *tone* and *emotional content* of text. This is crucial because biased language is often characterized by strong, one-sided emotional framing.

### What we built:
- Fine-tuned a **BERT** and **RoBERTa** model on a Reddit dataset annotated with "Us vs. Them" bias scores.
- Used **Multi-Task Learning (MTL)**: the model simultaneously predicts:
  - A **bias score** (continuous float, 0.0–1.0) — how much "Us vs. Them" framing is present.
  - **13 emotion categories** (Anger, Contempt, Disgust, Fear, Gratitude, etc.) — multi-label classification.
  - The **social group** being targeted (e.g., Refugees, Immigrants, Liberals).
- The Transformer backbone is **forked at the last layer**: the first 11 layers are shared across all tasks, and the 12th layer is independently duplicated per task. This lets each head specialize while still sharing a common language understanding.
- Trained with **GradNorm**, a technique that dynamically re-weights the loss of each task during backpropagation to prevent any single task from dominating the gradient signal.
- Emotion model checkpoints and threshold optimization live in `EmotionModels/`.

---

## Stage 2 — Graph Neural Network (Semi-Supervised Learning)

**Location:** `GraphNeuralNetwork/`

Raw text classification has limits — an article's bias is not just a function of its words, but also of *who wrote it*, *what outlet published it*, and *how it relates to other articles*. Stage 2 builds a knowledge graph of these relationships and uses a **Graph Neural Network (GNN)** to propagate bias labels across the graph in a semi-supervised fashion.

### What we built:
- **Article Embeddings**: Each article is embedded using the fine-tuned Transformer from Stage 1, capturing its semantic content and emotional tone.
- **Graph Construction**: A graph is built where nodes are articles, authors, and outlets. Edges represent relationships (same author, same topic, same outlet, shared entities, etc.).
- **Semi-Supervised GNN**: Only a fraction of articles are labeled. The GNN propagates these labels to unlabeled nodes by learning from the graph structure — articles connected to known biased sources are likely biased themselves.
- This approach dramatically reduces the need for hand-labeled data, which is expensive and time-consuming in the media domain.

---

## Stage 3 — Media Bias Prediction Model

**Location:** `SentimentClassification/` (final prediction layer)

The outputs of Stages 1 and 2 are combined into a final **Media Bias Prediction Model** that produces a unified, interpretable bias score for any given news article.

### What we built:
- Aggregates the emotion scores, "Us vs. Them" regression output, and GNN-propagated bias signal.
- Outputs a final **bias label** (e.g., Left / Center / Right, or a continuous bias score).
- Exposes this model as a **FastAPI REST API** so any client — web, mobile, or desktop — can query it in real time.

---

## Stage 4 — FastAPI Backend + C# MAUI Frontend

**Location:** `FrontEnd/MediaBiasApp/`

The final stage brings everything together in a polished, cross-platform user experience.

### FastAPI Backend
- Wraps the trained bias prediction model in a **FastAPI** REST endpoint.
- Accepts a news article URL or text body as input and returns:
  - A bias score.
  - A breakdown of detected emotions.
  - The predicted political leaning.
- Designed to be lightweight and deployable as a Docker container or cloud function.

### C# MAUI Frontend
- A **cross-platform app** built with .NET MAUI, targeting Windows, Android, and iOS from a single codebase.
- Integrates with a **News API** to fetch live, real-time news headlines and articles.
- Users can browse the latest news, select an article, and trigger a bias analysis query against the FastAPI backend.
- Results are displayed with intuitive visualizations (bias meters, emotion breakdowns, source credibility ratings).

---

## Repository Structure

```
media-bias-pipeline/
|
├── SentimentClassification/     # BERT/RoBERTa training pipeline
│   ├── train.py                 # Entry point — CLI-driven training
│   ├── dataloader.py            # Dataset + tokenization collator
│   ├── BertRegression.py        # BERT Lightning module
│   ├── RoBERTaRegression.py     # RoBERTa Lightning module
│   ├── RedditTransformer.py     # Custom forked BERT architecture
│   └── RoBERTaMTL.py            # Custom forked RoBERTa architecture
|
├── EmotionModels/               # Emotion classification & threshold tuning
|
├── GraphNeuralNetwork/          # GNN models, graph construction, embeddings
|
├── FrontEnd/
│   └── MediaBiasApp/            # .NET MAUI cross-platform app (C#)
|
├── utils/                       # Shared utilities
├── tb_logs/                     # TensorBoard training logs
├── MediaBiasPipeline.slnx       # Visual Studio solution file
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| NLP Models | Python, PyTorch, PyTorch Lightning, HuggingFace Transformers |
| Graph Learning | PyTorch Geometric / DGL |
| API Backend | FastAPI, Uvicorn |
| Frontend | .NET MAUI (C#), News API |
| Monitoring | TensorBoard |
| Database | PostgreSQL |
| Deployment | Docker, WSL2 |

---

## Further Reading

- [Project Overview](PROJECT_OVERVIEW.md) — Dataset schema, model breakdown, and CLI flags.
- [Code Walkthrough](CODE_WALKTHROUGH.md) — Deep-dive implementation guide for the NLP pipeline.