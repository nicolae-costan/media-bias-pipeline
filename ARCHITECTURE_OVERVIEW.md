# Architecture Overview — Media Bias Pipeline

---

## What is an Embedding?

An embedding is **not** a score. It is a **dense numerical fingerprint** of the meaning of a piece of text.

When BERT reads the sentence *"Immigrants are invading our country"*, it does not output a single number like `0.8 bias`. Instead, it outputs a **vector of 768 numbers** — e.g. `[0.21, -0.54, 0.87, 0.03, ...]`. These 768 numbers together encode the **semantic meaning** of that sentence in a mathematical space.

The magic: sentences with **similar meanings end up close together** in that 768-dimensional space, even if they use different words. *"Immigrants are invading"* and *"Foreigners are taking over"* would have very similar embedding vectors. That is what makes them useful for the graph — you connect articles that are semantically similar.

| Term | What it is |
|---|---|
| **Embedding** | A vector of 768 floats — the "meaning" of the text in math space |
| **Emotion scores** | A vector of 13 floats (one per emotion, 0–1 probability) — output of the Emotion Model's head |
| **Bias score** | A single float (0–1) — output of the Bias Model's regression head |

---

## What is BABE?

**BABE = Bias Annotations By Experts.** It is a dataset of ~3,700 news article sentences (not Reddit comments) that were manually annotated by expert journalists and researchers for media bias.

In this project it lives at `GraphNeuralNetwork/data/final_labels_MBIC.csv`. It has the following columns:

| Column | Description |
|---|---|
| `text` | The sentence from the news article |
| `outlet` | e.g. `"foxnews"`, `"nytimes"`, `"thefederalist"` |
| `topic` | e.g. `"abortion"`, `"immigration"` |
| `type` | Political leaning: `"left"`, `"right"`, `"center"` |
| `article` | Full text of the original article |
| `news_link` | URL of the source article |
| `label_bias` | **`"Biased"` or `"Non-biased"`** — the ground truth label |
| `article_id` | A unique hash ID for the article |

BABE is the **labelled dataset for the Graph Neural Network**. The GNN uses these labels to know which articles are biased and which are not, then learns to classify unlabelled ones by looking at their neighbourhood in the graph.

---

## Module Map: What Each Part Does

```
media-bias-pipeline/
├── SentimentClassification/   ← BIAS MODEL (Reddit, Us-vs-Them scoring)
│   ├── RedditTransformer.py   (BERT neural network architecture)
│   ├── BertRegression.py      (BERT training harness — PyTorch Lightning)
│   ├── RoBERTaMTL.py          (RoBERTa architecture — more advanced, 3-branch MTL)
│   ├── RoBERTaRegression.py   (RoBERTa training harness)
│   ├── train.py               (entry point: parse args, build Trainer, call fit)
│   └── test.py                (load checkpoint, run on test set, write predictions.csv)
│
├── EmotionModels/             ← EMOTION MODEL (news articles, 13-label classification)
│   ├── model.py               (EmotionModel class — fine-tuned transformer)
│   ├── train.py               (training entry point)
│   ├── test.py                (evaluation + threshold loading)
│   ├── optimize_thresholds.py (per-class threshold search for best Jaccard score)
│   ├── dataloader.py          (dataset + collator for emotion labels)
│   └── visualize_emotions.py  (plots: distribution, heatmap, t-SNE, word clouds)
│
└── GraphNeuralNetwork/        ← GNN (uses output of both models above)
    ├── article_embeddings.py  (runs Emotion Model → extracts embeddings → stores them)
    ├── build_graph.py         (builds k-NN graph from embeddings, reads BABE labels)
    ├── GraphModel.py          (Graph Attention Network architecture)
    ├── train.py               (trains the GAT on labelled nodes)
    └── test.py                (evaluates GAT on test nodes)
```

---

## The Three Models Explained

### 1. The Bias Model (`SentimentClassification/`)

- **Input**: A Reddit comment (raw text)
- **Output**: A float 0–1 called `usVSthem_scale` — how "Us vs. Them" the comment is
- **Architecture**: BERT or RoBERTa. The last transformer layer is **forked into two branches**: one specialises in the bias regression, the other in an auxiliary task (emotions or social group). This is multi-task learning.
- **Dataset**: `UsVsThem_train/valid/test_public.csv` — Reddit comments labelled by humans on a 0–1 scale
- **Role in the pipeline**: Standalone model. Its output is a score for Reddit text. It does not feed into the GNN.

### 2. The Emotion Model (`EmotionModels/`)

- **Input**: A news article sentence
- **Output**: 13 probability scores (Anger, Contempt, Disgust, Fear, Gratitude, Guilt, Happiness, Hope, Pride, Relief, Sadness, Sympathy, Neutral)
- **Architecture**: A transformer fine-tuned using Asymmetric Loss (to handle extreme class imbalance) with per-class threshold optimisation. Uses **Mean Pooling** (MInT) over all tokens instead of only the `[CLS]` token.
- **Dataset**: BABE/MBIC data with emotion labels
- **Role in the pipeline**: Its trained checkpoint is **loaded by `article_embeddings.py`**. That script passes each of the 13,697 articles through this model to extract both the 768-dim embedding (from the encoder's last hidden state) and the 13 emotion scores. These are then stored in the database.

### 3. The Graph Neural Network (`GraphNeuralNetwork/`)

- **Input**: The embeddings + emotion scores produced by the Emotion Model; the bias labels from BABE
- **Output**: A `Biased` / `Non-biased` prediction for each article node in the graph
- **Architecture**: Graph Attention Network (GAT). Nodes = articles. Edges = pairs of articles with cosine similarity above a threshold. Edge weights are blended from embedding similarity (80%) and emotion similarity (20%).
- **Dataset**: BABE for labels; `merged_clean_data.csv` for the full 13k article pool
- **Role in the pipeline**: The final classification stage for news articles. It leverages the graph structure — an article surrounded by many biased neighbours is more likely to be biased itself.

---

## The Full Pipeline, End to End

```
Step 1 — Train Emotion Model
  EmotionModels/train.py
  → Fine-tunes a transformer on BABE emotion labels
  → Saves a checkpoint: tb_logs/emotion_classification/version_N/checkpoints/*.ckpt

Step 2 — Generate Embeddings
  GraphNeuralNetwork/article_embeddings.py  (originally via PySpark)
  → Loads the Emotion Model checkpoint
  → Reads merged_clean_data.csv (13,697 articles)
  → For each article:
      - Runs BERT encoder → 768-dim embedding (mean pooling over last hidden state)
      - Runs classifier head → 13 emotion probability scores
  → Stores (article_id, embedding[768], emotion_scores[13]) in the database
     Originally: PostgreSQL with pgvector
     Current branch: ChromaDB in ./vector_db/

Step 3 — Build the Graph
  GraphNeuralNetwork/build_graph.py
  → Reads all embeddings from the database
  → Reads BABE labels (Biased / Non-biased) from final_labels_MBIC.csv
  → Computes annotator agreement from SG1/SG2 CSVs → decides train/val/test masks
  → Builds k-NN graph:
      - For each article, find top-K most similar neighbours (cosine similarity)
      - If similarity ≥ threshold → add edge
      - Adjust edge weight: +bonus if same label, −penalty if different label
  → Saves graph.pt (PyTorch Geometric Data object)

Step 4 — Train the GNN
  GraphNeuralNetwork/train.py
  → Loads graph.pt
  → Trains Graph Attention Network on nodes where train_mask=True (high annotator agreement)
  → Validates on val_mask nodes, tests on test_mask nodes
  → Uses message-passing: each node aggregates information from its neighbours

Step 5 — Bias Model (separate track, independent of GNN)
  SentimentClassification/train.py
  → Trains BERT or RoBERTa on Reddit UsVsThem data
  → Outputs a continuous 0–1 bias score for any Reddit comment
  → Evaluated on test.py → writes predictions.csv
```

---

## Why Does `SentimentClassification` Exist If the GNN Doesn't Use It?

`SentimentClassification/` is the **original, standalone project**. The GNN was added later as a new research direction. They currently run on completely different data and **do not connect to each other**.

| | SentimentClassification | GraphNeuralNetwork |
|---|---|---|
| **Data source** | Reddit comments | News articles (BABE) |
| **What it labels** | User-generated social media text | Professional journalism |
| **Label type** | Continuous 0–1 "Us vs Them" scale | Binary: Biased / Non-biased |
| **Label origin** | Crowd-sourced human annotation | Expert journalist annotation |
| **Output use** | `predictions.csv` | Graph node labels |

They tackle **two different definitions of "bias"** on **two different kinds of text**.

`SentimentClassification` was the first thing built — the EmotionModel and GNN were added on top as a new direction. It remains a working, independent model for scoring Reddit comments for polarised "us vs them" language.

### Could they be connected in the future?

Yes, theoretically. A few ways this could work:

1. **Reddit comments linked to news articles** → run `SentimentClassification` on those comments and use the scores as additional node features in the graph.
2. **Cross-signal** → "this news article generated highly polarised Reddit comments → extra evidence of bias".
3. **Ensemble** → an article is more likely biased if both the GNN *and* the `SentimentClassification` model agree.

None of this integration is implemented yet. Right now they are genuinely parallel tracks.
