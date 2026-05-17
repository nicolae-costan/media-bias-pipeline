# Technical Architecture Overview — Media Bias Pipeline

This document details the software architecture, data models, and machine learning components of the Media Bias Analysis pipeline. 

---

## 1. Core Data Abstractions

### Dense Text Embeddings
Embeddings are high-dimensional, continuous representations of text semantics. Raw input sequences are projected into a 768-dimensional vector space using the last hidden states of a pre-trained transformer model (e.g., RoBERTa/BERT). 

*   **Extraction Method:** Mean pooling (Mean-in-Transformer / MInT pooling) is computed over all non-padding tokens in the sequence, ensuring representation of the entire text context rather than relying solely on the class token (`[CLS]`).
*   **Vector Dimensions:** 768 floats (FP32).
*   **Downstream Usage:** The Graph Neural Network (GNN) utilizes cosine similarity metrics between these vectors to establish topological connections (edges) in the KNN semantic graph.

### Feature Definitions
*   **Embedding Vector:** A 768-dimensional dense numerical fingerprint representing semantic content.
*   **Emotion Probability Vector:** A 13-dimensional vector of floats representing the predicted probability (0.0 to 1.0) of 13 granular emotions (Anger, Contempt, Disgust, Fear, Gratitude, Guilt, Happiness, Hope, Pride, Relief, Sadness, Sympathy, Neutral).
*   **Consensus Label (`label_bias`):** Ground-truth binary classification representing consensus annotator label ("Biased" or "Non-biased"), computed via Majority Voting on raw study groups.
*   **Certainty Metric (`agreement`):** The ratio of consensus annotations over total annotations per article, serving as a sample confidence weight during GNN training.

---

## 2. Dataset Glossary

### BABE (Bias Annotations By Experts)
The core dataset utilized for the Graph Neural Network pipeline, consisting of news article sentences labeled by expert journalists.
*   **Primary Metadata:** Source outlet (e.g., Fox News, NYTimes), topic (e.g., abortion, immigration), political leaning (left/right/center), and source links.
*   **Raw Annotations (SG1 & SG2):** Separate study group labels representing the individual judgments of multiple annotators, used to compute majority vote consensus labels and agreement confidence metrics.

---

## 3. Modular System Mapping

```
media-bias-pipeline/
├── utils/                     ← Shared PySpark and processing routines
│   └── utils.py               (Consensus computing via PySpark engine)
│
├── SentimentClassification/   ← Baseline Multi-Task Bias Regression (Reddit Track)
│   ├── RedditTransformer.py   (Custom dual-head BERT architecture)
│   ├── BertRegression.py      (PyTorch Lightning regression model)
│   ├── RoBERTaMTL.py          (Multi-Task Learning architecture with 3-branch split)
│   ├── RoBERTaRegression.py   (RoBERTa regression training harness)
│   ├── train.py               (Training entry point)
│   └── test.py                (Evaluation entry point, outputs predictions.csv)
│
├── EmotionModels/             ← Emotion Classifier (News Track)
│   ├── model.py               (Asymmetric Loss fine-tuned Transformer)
│   ├── train.py               (Training entry point)
│   ├── test.py                (Evaluation + threshold inference)
│   ├── optimize_thresholds.py (Per-class F1/Jaccard threshold optimization)
│   └── dataloader.py          (Dataset and collator for multi-label data)
│
└── GraphNeuralNetwork/        ← Semi-Supervised Graph Attention Network (GAT)
    ├── article_embeddings.py  (Extracts embeddings & emotion features → stores in PG)
    ├── build_graph.py         (Constructs k-NN GNN graph with agreement filtering)
    ├── GraphModel.py          (GAT neural network layers)
    ├── train.py               (Trains GAT classifier using node masking)
    └── test.py                (Evaluates GAT performance on testing partitions)
```

---

## 4. Machine Learning Components

### A. The Baseline Bias Model (`SentimentClassification/`)
*   **Target Task:** Social media bias classification on Reddit text.
*   **Loss / Objective:** Mean Squared Error (MSE) regression over continuous "Us vs. Them" scale (0.0 to 1.0) along with multi-task aux heads.
*   **Architecture:** Multi-Task Transformer (BERT/RoBERTa) with shared lower-level representations and branched task-specific classification heads.
*   **Context in Project:** Serves as a **standalone experimental track and baseline reference**. It was developed to explore continuous regression and multi-task learning structures on social media text before the pipeline transitioned to relational GNN structures on news data.

### B. The Emotion Model (`EmotionModels/`)
*   **Target Task:** Granular emotion classification on news media text (13 labels).
*   **Loss / Objective:** Asymmetric Loss (ASL) to dynamically down-weight the loss contributions of negative samples in extremely imbalanced multi-label environments.
*   **Architecture:** Fine-tuned Transformer using MInT pooling and class-specific optimal thresholds.
*   **Role in Pipeline:** The trained checkpoint is utilized to generate the node features for the GNN. The BERT/RoBERTa body extracts the semantic embedding, and the classification head outputs the emotional distributions which are stored in the database.

### C. Graph Attention Network (`GraphNeuralNetwork/`)
*   **Target Task:** Semi-supervised binary classification ("Biased" / "Non-biased") of news articles.
*   **Architecture:** Graph Attention Network (GAT) using multi-head attention over node features.
*   **Graph Structure:** Node features consist of the 768-dim semantic embedding concatenated with the 13-dim emotion vector. Edges are generated dynamically using K-Nearest Neighbors based on cosine similarity, filtered by annotator agreement thresholds.

---

## 5. Unified Processing Pipeline

```
Phase 1: Feature Extraction & Preprocessing
  [Raw CSVs] ──> [utils/utils.py (PySpark Engine)] ──> [Consensus Labels + Agreement]
  [News Texts] ──> [Emotion Classifier Checkpoint] ──> [768-dim Vectors + 13-dim Emotion Scores]
                          │
                          ▼
             [PostgreSQL + pgvector Database]

Phase 2: Topological Graph Generation
  [Database] ──> [build_graph.py] ──> [KNN Cosine Graph + Annotator Confidence Masking] ──> [graph.pt]

Phase 3: GAT Model Training & Evaluation
  [graph.pt] ──> [GraphNeuralNetwork/train.py] ──> [GAT Binary Bias Predictor]
```

This multi-stage architecture leverages the representational power of Transformers for dense feature extraction, the distributed efficiency of PySpark for clean data wrangling, and the relational power of Graph Attention Networks to map structural media patterns.
