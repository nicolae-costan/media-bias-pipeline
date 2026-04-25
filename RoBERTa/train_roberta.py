"""
RoBERTa Multi-Label Emotion Classifier (v3 — Label Aggregation)
================================================================
Combats extreme class imbalance by aggregating 13 granular emotions
into 4 macro-level categories BEFORE training:

  Hostility     ← Anger, Contempt, Disgust, Fear
  Vulnerability ← Sadness, Guilt, Sympathy
  Positive      ← Happiness, Hope, Pride, Relief, Gratitude
  Neutral       ← Emotions_Neutral

Pipeline features retained from v2:
  1. Asymmetric Loss (replaces BCE)
  2. Per-class dynamic thresholding (replaces hard 0.5)
  3. MLP classification head (replaces single Linear)
  4. Differential learning rates + layer freezing
"""
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score, hamming_loss, jaccard_score
from tqdm import tqdm

# ============================================================================
# ORIGINAL 13 EMOTION COLUMNS — needed for reading the CSV
# ============================================================================
ORIGINAL_EMOTION_COLUMNS = [
    'Anger', 'Contempt', 'Disgust', 'Fear', 'Gratitude', 'Guilt',
    'Happiness', 'Hope', 'Pride', 'Relief', 'Sadness', 'Sympathy',
    'Emotions_Neutral'
]

# ============================================================================
# LABEL AGGREGATION MAP — 4 macro categories
# ============================================================================
MACRO_LABEL_MAP = {
    'Hostility':     ['Anger', 'Contempt', 'Disgust', 'Fear'],
    'Vulnerability': ['Sadness', 'Guilt', 'Sympathy'],
    'Positive':      ['Happiness', 'Hope', 'Pride', 'Relief', 'Gratitude'],
    'Neutral':       ['Emotions_Neutral'],
}

MACRO_COLUMNS = list(MACRO_LABEL_MAP.keys())  # deterministic order
NUM_CLASSES = len(MACRO_COLUMNS)  # 4


def aggregate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with the original 13 binary emotion columns,
    create 4 new macro-label columns using an OR (max) aggregation:
    the macro-label is 1 if ANY constituent granular emotion is 1.

    Returns the same DataFrame with the new columns appended.
    """
    for macro_name, granular_cols in MACRO_LABEL_MAP.items():
        df[macro_name] = df[granular_cols].max(axis=1).clip(upper=1).astype(int)
    return df


# ============================================================================
# 1. DATA SETUP
# ============================================================================

class EmotionDataset(Dataset):
    """
    Dataset for multi-label emotion classification with label aggregation.
    Reads a CSV with a 'body' text column and 13 binary emotion columns,
    then aggregates them into 4 macro-level labels on the fly.
    """
    def __init__(self, csv_path, tokenizer, max_length=128):
        self.df = pd.read_csv(csv_path)
        # --- Aggregate 13 → 4 macro labels ---
        self.df = aggregate_labels(self.df)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.iloc[idx]['body'])
        labels = self.df.iloc[idx][MACRO_COLUMNS].values.astype(np.float32)

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids':      encoding['input_ids'].flatten(),
            'attention_mask':  encoding['attention_mask'].flatten(),
            'labels':          torch.tensor(labels, dtype=torch.float)
        }


def calculate_pos_weights(csv_path):
    """
    pos_weight_j = num_negatives_j / num_positives_j
    Computed over the 4 aggregated macro labels.
    Used for diagnostic printing; the Asymmetric Loss handles
    imbalance internally.
    """
    df = pd.read_csv(csv_path)
    df = aggregate_labels(df)
    labels = df[MACRO_COLUMNS].values
    pos = labels.sum(axis=0)
    neg = len(df) - pos
    weights = neg / (pos + 1e-6)
    return torch.tensor(weights, dtype=torch.float)


# ============================================================================
# 2. ASYMMETRIC LOSS  (replaces BCEWithLogitsLoss)
# ============================================================================
# Reference:  Emanuel Ben-Baruch et al., "Asymmetric Loss for Multi-Label
#             Classification", ICCV 2021.
#
# Core idea:  Two separate focusing parameters:
#   • gamma_pos  — down-weights *easy positives* (high-confidence correct 1s)
#   • gamma_neg  — down-weights *easy negatives* (high-confidence correct 0s)
#                  Setting gamma_neg > gamma_pos forces the model to focus on
#                  hard positives (rare classes it keeps missing).
#
# Probability shifting (clip):
#   Before computing the negative part of the loss we shift p_neg down by
#   `clip`, i.e.  p_neg_shifted = max(p_neg - clip, 0).
#   This acts as a hard-negative mining knob: anything the model is already
#   confident is negative gets zero gradient, freeing capacity for positives.
# ============================================================================

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification.

    Args:
        gamma_neg (float): Focusing parameter for negatives  (default 4).
        gamma_pos (float): Focusing parameter for positives  (default 1).
        clip      (float): Probability-margin clipping for negatives (default 0.05).
        eps       (float): Label smoothing epsilon            (default 0.1).
    """
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=0.1):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        # --- Probabilities via sigmoid ---
        probs = torch.sigmoid(logits)
        # Numerical safety
        probs = probs.clamp(min=1e-7, max=1 - 1e-7)

        # --- Separate positive / negative probabilities ---
        probs_pos = probs          # p   when y=1
        probs_neg = 1.0 - probs    # 1-p when y=0

        # --- Probability shifting (hard-negative mining for negatives) ---
        if self.clip > 0:
            # Shift the negative probabilities DOWN so easy negatives → 0 loss
            probs_neg = (probs_neg + self.clip).clamp(max=1.0)

        # --- Basic cross-entropy components ---
        loss_pos = -targets * torch.log(probs_pos)
        loss_neg = -(1 - targets) * torch.log(probs_neg)

        # --- Focal modulation ---
        if self.gamma_pos > 0:
            # Down-weight easy positives: the model is already confident
            focal_weight_pos = (1.0 - probs_pos) ** self.gamma_pos
            loss_pos = loss_pos * focal_weight_pos

        if self.gamma_neg > 0:
            # Aggressively down-weight easy negatives
            focal_weight_neg = probs ** self.gamma_neg   # probs (not probs_neg)
            loss_neg = loss_neg * focal_weight_neg

        # --- Label smoothing ---
        if self.eps > 0:
            loss_pos = loss_pos * (1 - self.eps) + self.eps * loss_neg
            loss_neg = loss_neg * (1 - self.eps) + self.eps * loss_pos

        loss = loss_pos + loss_neg
        return loss.mean()


# ============================================================================
# 3. MODEL — RoBERTa + MLP Head + Layer Freezing
# ============================================================================

class RoBERTaEmotionClassifier(nn.Module):
    """
    RoBERTa-base encoder with:
      • First 8 transformer layers FROZEN (retain linguistic priors)
      • Last 4 layers + pooler fine-tuned with a small LR
      • A 2-layer MLP classification head trained with a larger LR

    Output: 4 macro-category logits (no sigmoid).
    """
    def __init__(self, n_classes=NUM_CLASSES, head_dropout=0.2):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        hidden = self.roberta.config.hidden_size  # 768

        # --- MLP classification head ---
        # Linear(768→768) → GELU → Dropout → Linear(768→4)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden, n_classes),
        )

        # Xavier init for the head so it starts in a good region
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        # --- Freeze first 8 encoder layers ---
        self._freeze_layers(n_freeze=8)

    # -----------------------------------------------------------------
    def _freeze_layers(self, n_freeze=8):
        """Freeze embeddings + first `n_freeze` transformer layers."""
        # Freeze embeddings
        for p in self.roberta.embeddings.parameters():
            p.requires_grad = False

        # Freeze layers 0 .. n_freeze-1
        for layer_idx in range(n_freeze):
            for p in self.roberta.encoder.layer[layer_idx].parameters():
                p.requires_grad = False

        # Diagnostic: count trainable params
        total   = sum(p.numel() for p in self.parameters())
        frozen  = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f"[Model] Total params: {total:,}  |  "
              f"Frozen: {frozen:,}  |  Trainable: {total - frozen:,}")

    # -----------------------------------------------------------------
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids,
                               attention_mask=attention_mask)
        pooled = outputs.pooler_output          # [B, 768]
        logits = self.classifier(pooled)        # [B, 4]
        return logits

    # -----------------------------------------------------------------
    def get_param_groups(self, lr_encoder=1e-5, lr_head=5e-4,
                         weight_decay=0.01):
        """
        Build two optimizer param groups:
          1. Unfrozen RoBERTa layers  → small LR  (lr_encoder)
          2. MLP classification head  → large LR  (lr_head)
        """
        encoder_params = [
            p for n, p in self.roberta.named_parameters() if p.requires_grad
        ]
        head_params = list(self.classifier.parameters())

        return [
            {'params': encoder_params,
             'lr': lr_encoder,
             'weight_decay': weight_decay},
            {'params': head_params,
             'lr': lr_head,
             'weight_decay': weight_decay},
        ]


# ============================================================================
# 4. PER-CLASS DYNAMIC THRESHOLD SEARCH
# ============================================================================

def find_optimal_thresholds(all_probs, all_targets, grid_start=0.1,
                            grid_end=0.9, grid_step=0.05):
    """
    For EACH of the 4 macro emotion classes independently, sweep over a
    range of thresholds and pick the one that maximises per-class F1.

    Args:
        all_probs:   np.ndarray  [N, 4]  sigmoid probabilities
        all_targets: np.ndarray  [N, 4]  ground-truth multi-hot

    Returns:
        best_thresholds: np.ndarray [4]
    """
    n_classes = all_targets.shape[1]
    thresholds = np.arange(grid_start, grid_end + 1e-9, grid_step)
    best_thresholds = np.full(n_classes, 0.5)

    for cls_idx in range(n_classes):
        best_f1 = -1.0
        y_true = all_targets[:, cls_idx]

        for t in thresholds:
            y_pred = (all_probs[:, cls_idx] >= t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresholds[cls_idx] = t

    return best_thresholds


# ============================================================================
# 5. EVALUATION
# ============================================================================

def evaluate(model, data_loader, device, criterion, thresholds=None):
    """
    Runs the model on `data_loader` and computes:
      • Loss (Asymmetric)
      • Macro / Micro F1
      • Hamming Loss
      • Per-class F1 for each of the 4 macro categories

    If `thresholds` is None, we first run a threshold search on the same data
    and then report metrics with the optimal thresholds.  Otherwise the
    supplied thresholds are reused (e.g. applying val thresholds to test).

    Returns:
        metrics    (dict)       — aggregated scores
        thresholds (np.ndarray) — the per-class thresholds that were used
    """
    model.eval()
    losses = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)
            losses.append(loss.item())

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(labels.cpu().numpy())

    all_probs   = np.concatenate(all_probs,   axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # --- Dynamic threshold search (per-class) ---
    if thresholds is None:
        thresholds = find_optimal_thresholds(all_probs, all_targets)
        print(f"  Optimal thresholds per class:")
        for name, t in zip(MACRO_COLUMNS, thresholds):
            print(f"    {name:20s}  →  {t:.2f}")

    # --- Apply thresholds ---
    all_preds = (all_probs >= thresholds[np.newaxis, :]).astype(int)

    metrics = {
        'loss':           np.mean(losses),
        'macro_f1':       f1_score(all_targets, all_preds, average='macro',
                                   zero_division=0),
        'micro_f1':       f1_score(all_targets, all_preds, average='micro',
                                   zero_division=0),
        'hamming_loss':   hamming_loss(all_targets, all_preds),
    }

    # --- Per-class F1 breakdown for the 4 macro categories ---
    per_class_f1 = f1_score(all_targets, all_preds, average=None,
                            zero_division=0)
    print("  Per-class F1:")
    for name, f in zip(MACRO_COLUMNS, per_class_f1):
        print(f"    {name:20s}  →  {f:.4f}")

    return metrics, thresholds


# ============================================================================
# 6. TRAINING LOOP
# ============================================================================

def train():
    # ----- Configuration -----
    TRAIN_CSV       = 'Resources/UsVsThem_train_public.csv'
    VAL_CSV         = 'Resources/UsVsThem_valid_public.csv'
    BATCH_SIZE      = 16
    MAX_LEN         = 128
    EPOCHS          = 5
    LR_ENCODER      = 1e-5       # small LR for unfrozen RoBERTa layers
    LR_HEAD         = 5e-4       # larger LR for the MLP head
    WEIGHT_DECAY    = 0.01
    PATIENCE        = 3          # more patience — ASL converges slower than BCE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Config] Device: {DEVICE}")

    # ----- Tokenizer -----
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # ----- Datasets & Loaders -----
    train_dataset = EmotionDataset(TRAIN_CSV, tokenizer, MAX_LEN)
    val_dataset   = EmotionDataset(VAL_CSV,   tokenizer, MAX_LEN)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                               shuffle=True,  num_workers=2, pin_memory=True)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=2, pin_memory=True)

    # ----- Diagnostic: class distribution (aggregated) -----
    pos_weights = calculate_pos_weights(TRAIN_CSV)
    print(f"\n[Data] Aggregated macro-label pos_weights (for reference):")
    for name, w in zip(MACRO_COLUMNS, pos_weights):
        print(f"  {name:20s}  →  {w:.2f}")

    # ----- Model -----
    model = RoBERTaEmotionClassifier(n_classes=NUM_CLASSES).to(DEVICE)

    # ----- Asymmetric Loss -----
    # gamma_neg=4 aggressively down-weights easy negatives;
    # gamma_pos=1 keeps the loss sensitive to hard positives.
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05, eps=0.1)

    # ----- Optimizer with differential LRs -----
    param_groups = model.get_param_groups(lr_encoder=LR_ENCODER,
                                          lr_head=LR_HEAD,
                                          weight_decay=WEIGHT_DECAY)
    optimizer = AdamW(param_groups)

    # ----- Scheduler (linear warmup) -----
    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ----- Training state -----
    best_macro_f1 = -1.0        # track *Macro F1* now, not loss
    trigger_times = 0
    best_thresholds = None

    for epoch in range(EPOCHS):
        # ---- Train phase ----
        model.train()
        train_losses = []
        print(f"\n{'='*60}")
        print(f"  EPOCH {epoch+1} / {EPOCHS}")
        print(f"{'='*60}")

        loop = tqdm(train_loader, leave=True, desc="Training")
        for batch in loop:
            optimizer.zero_grad()

            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels         = batch['labels'].to(DEVICE)

            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            loop.set_postfix(loss=f"{np.mean(train_losses[-50:]):.4f}")

        avg_train_loss = np.mean(train_losses)
        print(f"\n  Avg Train Loss: {avg_train_loss:.4f}")

        # ---- Validation phase (with dynamic threshold search) ----
        print("\n  --- Validation ---")
        val_metrics, val_thresholds = evaluate(
            model, val_loader, DEVICE, criterion, thresholds=None
        )

        print(f"\n  Validation Results:")
        for k, v in val_metrics.items():
            print(f"    {k:18s}: {v:.4f}")

        # ---- Early stopping on Macro F1 (not loss!) ----
        current_macro_f1 = val_metrics['macro_f1']
        if current_macro_f1 > best_macro_f1:
            best_macro_f1 = current_macro_f1
            best_thresholds = val_thresholds
            trigger_times = 0
            torch.save(model.state_dict(), 'best_roberta_emotions.pt')
            np.save('best_thresholds.npy', best_thresholds)
            print(f"  ✓ New best Macro F1: {best_macro_f1:.4f}  — model saved!")
        else:
            trigger_times += 1
            print(f"  ✗ No improvement ({trigger_times}/{PATIENCE})")
            if trigger_times >= PATIENCE:
                print("\n  ⚠ Early stopping triggered.")
                break

    # ----- Final summary -----
    print(f"\n{'='*60}")
    print(f"  Training complete.  Best Macro F1: {best_macro_f1:.4f}")
    print(f"  Macro categories: {MACRO_COLUMNS}")
    print(f"  Saved: best_roberta_emotions.pt  +  best_thresholds.npy")
    print(f"{'='*60}")


# ============================================================================
if __name__ == "__main__":
    train()
