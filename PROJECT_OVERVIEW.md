# Media Bias Pipeline — Project Overview

## What This Project Does

This project trains a **transformer-based model** (BERT or RoBERTa) to detect **"Us vs. Them" media bias** in Reddit comments.
The model reads a comment and outputs a **continuous bias score (0.0 – 1.0)**.  
It can optionally learn a second task at the same time (**multi-task learning**): predicting either emotions or the targeted social group mentioned in the comment. This often improves the main task.

---

## Dataset

All CSVs live in `Resources/` and share the same column schema:

| Column | Description |
|---|---|
| `body` | Raw Reddit comment text |
| `usVSthem_scale` | **Primary label** — Us vs. Them bias score (float, 0–1) |
| `bias` | Whether the comment contains bias at all |
| `group` | The social group being referenced |
| `Anger`, `Contempt`, … (×13) | Emotion labels (one-hot multi-label) |

| File | Role |
|---|---|
| `UsVsThem_train_public.csv` | Training set |
| `UsVsThem_valid_public.csv` | Validation set (checked after every epoch) |
| `UsVsThem_test_public.csv` | Test set (evaluated once at the very end) |
| `sample_train/dev/test.csv` | Tiny samples in the root — useful for quick smoke tests |

---

## File-by-File Breakdown

### `SentimentClassification/train.py` — **Entry Point**

> **Run with:** `python3 SentimentClassification/train.py --model_type bert --encoder_model bert-base-uncased --batch_size 8 --gpus 1 --max_epochs 10`

- Parses all CLI arguments (model type, batch size, epochs, paths, etc.) via `HyperOptArgumentParser`.
- Instantiates either `BERTRegressor` or `RoBERTaRegressor` based on `--model_type`.
- Sets up a **TensorBoard logger** (logs land in `tb_logs/<task>_<model>/version_N/`).
- Registers two callbacks: **EarlyStopping** (on `val_loss`, patience=10) and **ModelCheckpoint** (saves best `val_loss`).
- Hands everything off to PyTorch Lightning's `Trainer` and calls `trainer.fit()` → `trainer.test()`.
- Supports **hyperparameter search mode** (`--search_mode True`) which runs multiple trials.

---

### `SentimentClassification/dataloader.py` — **Data Pipeline**

Three things live here:

| Class / Function | What it does |
|---|---|
| `RedditDataset` | A `torch.utils.data.Dataset`. Reads a CSV, handles the aux task label differently depending on its type (`None`, `emotions`, or a class column like `group`). `__getitem__` returns one row: text + labels. |
| `MyCollator` | A callable collator passed to `DataLoader`. Strips URLs from comment text, tokenises with `AutoTokenizer` (padding + truncation to `max_length`), and bundles everything into `(tokenized_dict, targets_dict)` batches ready for the model. |
| `sentiment_analysis_dataset()` | Convenience wrapper — picks train/val/test based on flags and constructs a `RedditDataset`. |

---

### `SentimentClassification/BertRegression.py` — **BERT Lightning Module**

`BERTRegressor(pl.LightningModule)` is the training harness for the BERT backbone.

Key responsibilities:

| Method | Purpose |
|---|---|
| `__build_model()` | Reads all CSVs, fits `LabelEncoder`s, then instantiates `RedditTransformer` with the right head sizes. |
| `__build_loss()` | `MSELoss` for the main regression; `BCEWithLogitsLoss` (emotions) or `CrossEntropyLoss` (group) for the auxiliary head. |
| `forward()` | Delegates to `RedditTransformer`, packages outputs as a dict. |
| `training_step()` | Forward pass → compute loss → log `train_loss`. |
| `validation_step()` / `on_validation_epoch_end()` | Computes val loss, Pearson correlation, aux accuracy, confusion matrix. Logs everything + adds confusion matrix figure to TensorBoard. |
| `test_step()` / `on_test_epoch_end()` | Same as validation but also **saves `predictions.csv`** next to the checkpoint. |
| `backward()` | Custom backward with optional **GradNorm** — dynamically re-weights main vs. aux loss so neither dominates. |
| `configure_optimizers()` | Adam optimizer with a linear warmup LR scheduler. Encoder and aux head can get different learning rates. |
| `freeze_encoder()` / `unfreeze_encoder()` | Freezes the BERT encoder for `nr_frozen_epochs` epochs (only the last layer stays trainable), then unfreezes for fine-tuning. |
| `add_model_specific_args()` | Registers all BERT-specific CLI flags (`--encoder_model`, `--aux_task`, `--gradnorm`, `--nr_frozen_epochs`, data paths, etc.). |

---

### `SentimentClassification/RedditTransformer.py` — **Custom BERT Architecture**

This is the neural network itself, used by the BERT path.

| Class | What it does |
|---|---|
| `RedditTransformer` | Loads a pretrained `AutoModel` (BERT). If an aux task is active, it **replaces the last transformer layer** with two copies (`layer_main`, `layer_aux`) and two separate poolers — one for each task. Each stream then gets its own classification head. If no aux task, it's a standard single-head regressor. |
| `BertEncoder` | Custom encoder that runs layers 1–11 normally, then **forks into two independent 12th layers** (one per task). Returns a tuple of two hidden-state tensors. |
| `BertPooler` | Custom pooler that accepts the forked hidden states and applies a **separate linear + Tanh per task** on the `[CLS]` token. |

**Why the fork?** The idea is that lower layers learn shared language features, while the very last layer specialises per task — this is a classic multi-task learning trick.

---

### `SentimentClassification/RoBERTaRegression.py` — **RoBERTa Lightning Module**

`RoBERTaRegressor(pl.LightningModule)` is the training harness for the RoBERTa backbone. Same structural role as `BertRegression.py` but wraps `RoBERTaMTL` instead. A few differences:

- Cleaner **smart routing**: labels are explicitly routed to the correct head (`labels_emotion`, `labels_social`, `labels_bias`) based on `--aux_task`.
- Supports an **ablation mode** (`--ablate_social_group`) to zero-out the social head without removing it.
- Uses **mean pooling** over all tokens instead of just the `[CLS]` token.
- Simpler optimizer setup (single learning rate for all params).

---

### `SentimentClassification/RoBERTaMTL.py` — **Custom RoBERTa Architecture**

`RoBERTaMTL(nn.Module)` is the neural network for the RoBERTa path.

- Loads `roberta-base`, **truncates it to the first 11 layers**.
- **Deep-copies layer 12 three times** into `layer_bias`, `layer_emotion`, `layer_social` — one branch per task.
- Each branch feeds into its own head:
  - `bias_head` → Linear → Tanh → Dropout → Linear → **Sigmoid** (regression, 0-1 output)
  - `emotion_head` → Linear → Tanh → Dropout → Linear (logits for 13 emotions, BCEWithLogitsLoss)
  - `social_group_head` → Linear → Tanh → Dropout → Linear (logits for N social groups, CrossEntropyLoss)
- Loss computation happens **inside** `forward()` if labels are supplied; losses are weighted by `loss_weights`.
- `set_ablation_mode()` freezes and zeroes the social group branch for ablation studies.

---

### `SentimentClassification/test.py` — **Standalone Test Runner**

A separate script to re-run the **test set on an already-trained checkpoint** without retraining.

- Accepts `--checkpoint_path` pointing to the folder created by a training run.
- Scans that folder for a `.ckpt` file and loads the model.
- Spins up a `Trainer` in test-only mode and calls `trainer.test(model)`.

---

## End-to-End Training Workflow

```
python3 SentimentClassification/train.py \
    --model_type bert \
    --encoder_model bert-base-uncased \
    --batch_size 8 \
    --gpus 1 \
    --max_epochs 10
```

```
train.py  (parses args, builds Trainer)
   │
   ├─► BERTRegressor.__init__()
   │      ├─► __build_model()   ← reads CSVs, fits LabelEncoders
   │      │       └─► RedditTransformer(bert-base-uncased, ...)
   │      │               ├─► loads pretrained BERT weights
   │      │               ├─► (if aux task) replaces layer 12 with BertEncoder fork
   │      │               └─► adds classification head(s)
   │      └─► __build_loss()   ← MSELoss + optional aux loss
   │
   ├─► TensorBoardLogger  →  tb_logs/task_<aux>_bert/version_N/
   ├─► EarlyStopping (val_loss, patience=10)
   └─► ModelCheckpoint  →  saves best .ckpt to tb_logs/.../checkpoints/
   
   trainer.fit(model)
   │
   │  Per epoch:
   │    train_dataloader()  ←  RedditDataset(train CSV)  +  MyCollator
   │       └─► training_step()  →  forward()  →  loss()  →  backward()
   │    val_dataloader()    ←  RedditDataset(valid CSV)  +  MyCollator
   │       └─► validation_step()  →  on_validation_epoch_end()
   │              logs: val_loss, val_pearson, val_acc_aux, confusion matrix
   │
   trainer.test()
       test_dataloader()   ←  RedditDataset(test CSV)  +  MyCollator
          └─► test_step()  →  on_test_epoch_end()
                 logs: test_loss, test_pearson
                 writes: tb_logs/.../checkpoints/predictions.csv
```

---

## Key Hyperparameters (CLI flags)

| Flag | Default | Meaning |
|---|---|---|
| `--model_type` | `bert` | `bert` or `roberta` |
| `--encoder_model` | `bert-base-uncased` | HuggingFace model id |
| `--aux_task` | `None` | Secondary task: `None`, `bias`, `group`, `emotions` |
| `--batch_size` | `6` | Samples per GPU batch |
| `--gpus` | `1` | Number of GPUs (0 = CPU) |
| `--max_epochs` | `10` | Hard training cap |
| `--patience` | `10` | Early-stopping window |
| `--nr_frozen_epochs` | `0` | Epochs to freeze the encoder |
| `--gradnorm` | `False` | Enable GradNorm dynamic loss weighting |
| `--encoder_learning_rate` | `1e-5` | LR for the transformer weights |
| `--train_csv / --dev_csv / --test_csv` | `Resources/UsVsThem_*_public.csv` | Data file paths |

---

## Outputs After Training

| Path | Content |
|---|---|
| `tb_logs/<task>_<model>/version_N/` | TensorBoard event files (open with `tensorboard --logdir tb_logs`) |
| `tb_logs/.../checkpoints/<epoch>-<val_loss>.ckpt` | Best model checkpoint |
| `tb_logs/.../checkpoints/predictions.csv` | Two columns: `predicted_score, true_score` for every test sample |
