"""
optimize_thresholds.py
----------------------
Post-training step: run once after training finishes.

For each of the N emotion classes, grid-searches the decision threshold that
maximises the per-class Jaccard score on the validation set, then writes the
best thresholds to thresholds.json.

Usage (from inside EmotionModels/):
    python optimize_thresholds.py --checkpoint <path/to/checkpoint.ckpt>

The script auto-resolves the dev CSV from the checkpoint's saved hparams, so
you don't need to pass it manually unless you want to override it.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import jaccard_score, f1_score, classification_report
from torch.utils.data import DataLoader

# ── resolve imports whether run from project root or EmotionModels/ ─────────
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from model import EmotionModel
from dataloader import RedditDataset, sentiment_analysis_dataset, MyCollator


# ─────────────────────────────────────────────────────────────────────────────

def _resolve_path(path: str) -> str:
    """Try the path as-is, then relative to the script's parent directory."""
    if os.path.exists(path):
        return path
    alt = os.path.normpath(os.path.join(_script_dir, '..', path))
    if os.path.exists(alt):
        return alt
    return path  # let the caller handle missing file


def optimize_thresholds(checkpoint_path: str,
                        output_json: str = "thresholds.json",
                        dev_csv: str | None = None,
                        n_thresholds: int = 99):
    """
    1. Load model from checkpoint.
    2. Collect probabilities & targets on the validation set.
    3. Grid-search per-class thresholds on the first 50 % (calibration).
    4. Evaluate both baseline (0.5) and tuned thresholds on the second 50 %.
    5. Write optimal thresholds to *output_json*.
    """
    checkpoint_path = _resolve_path(checkpoint_path)
    print(f"\n{'='*60}")
    print(f"  Loading checkpoint: {checkpoint_path}")
    print(f"{'='*60}\n")

    # ── 1. Load model ────────────────────────────────────────────────────────
    model = EmotionModel.load_from_checkpoint(checkpoint_path, strict=False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Override dev CSV if explicitly requested
    if dev_csv is not None:
        model.hparams.dev_csv = dev_csv

    emotions = RedditDataset.EMOTION_COLUMNS
    num_emotions = len(emotions)
    print(f"  Emotion classes ({num_emotions}): {emotions}\n")

    # ── 2. Collect probabilities ─────────────────────────────────────────────
    print("Step A — Collecting probabilities on validation set …")
    dataset = sentiment_analysis_dataset(model.hparams, train=False, val=True, test=False)
    collator = MyCollator(model.hparams.encoder_model, model.hparams.max_length)
    loader = DataLoader(dataset, batch_size=model.hparams.batch_size,
                        collate_fn=collator, num_workers=0)

    all_probs, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="  inference"):
            input_ids, attention_mask = model._safe_squeeze(inputs)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            logits = model.forward(input_ids, attention_mask)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_targets.append(targets['labels_aux'].numpy())

    probs   = np.concatenate(all_probs,   axis=0)   # [N, num_emotions]
    targets = np.concatenate(all_targets, axis=0)   # [N, num_emotions]

    # ── 3. 50/50 calibration / internal-test split ──────────────────────────
    print("\nStep B — Splitting into calibration / internal-test halves …")
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(probs))
    split = len(probs) // 2
    calib_idx, test_idx = idx[:split], idx[split:]

    p_cal, t_cal = probs[calib_idx], targets[calib_idx]
    p_tst, t_tst = probs[test_idx],  targets[test_idx]

    # ── 4. Per-class grid search ─────────────────────────────────────────────
    print(f"\nStep C — Grid-searching {n_thresholds} thresholds per class …\n")
    thresholds = []
    header = f"  {'Emotion':<14} {'Best-T':>6}  {'Jaccard':>8}  {'F1':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for i, name in enumerate(emotions):
        y_true = t_cal[:, i]
        y_prob = p_cal[:, i]

        best_t, best_j = 0.5, -1.0
        for t in np.linspace(0.01, 0.99, n_thresholds):
            y_pred = (y_prob > t).astype(float)
            j = jaccard_score(y_true, y_pred, zero_division=0)
            if j > best_j:
                best_j, best_t = j, float(t)

        # Compute F1 at the chosen threshold for display
        y_pred_best = (y_prob > best_t).astype(float)
        f1 = f1_score(y_true, y_pred_best, zero_division=0)

        thresholds.append(best_t)
        print(f"  {name:<14} {best_t:>6.2f}  {best_j:>8.4f}  {f1:>8.4f}")

    # ── 5. Final evaluation ──────────────────────────────────────────────────
    print(f"\nStep D — Final evaluation on internal test set …\n")

    # Baseline @ 0.5
    base_preds = (p_tst > 0.5).astype(float)
    base_j = jaccard_score(t_tst, base_preds, average="macro", zero_division=0)
    base_f1 = f1_score(t_tst, base_preds, average="macro", zero_division=0)

    # Tuned thresholds
    tuned_preds = np.stack(
        [(p_tst[:, i] > thresholds[i]).astype(float) for i in range(num_emotions)],
        axis=1
    )
    tuned_j = jaccard_score(t_tst, tuned_preds, average="macro", zero_division=0)
    tuned_f1 = f1_score(t_tst, tuned_preds, average="macro", zero_division=0)

    print(f"  {'':20} {'Jaccard':>8}  {'Macro-F1':>9}")
    print(f"  {'Baseline (0.5)':20} {base_j:>8.4f}  {base_f1:>9.4f}")
    print(f"  {'Tuned':20} {tuned_j:>8.4f}  {tuned_f1:>9.4f}")
    print(f"  {'Improvement':20} {tuned_j - base_j:>+8.4f}  {tuned_f1 - base_f1:>+9.4f}")

    # Per-class report on tuned thresholds
    print(f"\n  Per-class report (tuned, internal test):\n")
    print(classification_report(
        t_tst, tuned_preds,
        target_names=emotions,
        zero_division=0,
        digits=4,
    ))

    # ── 6. Write thresholds ──────────────────────────────────────────────────
    out_path = os.path.join(_script_dir, output_json) if not os.path.isabs(output_json) else output_json
    with open(out_path, "w") as f:
        json.dump(thresholds, f, indent=4)
    print(f"\n  ✓ Thresholds saved to: {out_path}\n")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimise per-class decision thresholds")
    parser.add_argument(
        "--checkpoint", "-c", required=True,
        help="Path to the .ckpt file produced by train.py "
             "(e.g. tb_logs/emotion_classification/version_3/checkpoints/epoch=2-val_loss=0.0974.ckpt)"
    )
    parser.add_argument(
        "--dev_csv", default=None,
        help="Override the validation CSV path (default: use the path stored in the checkpoint hparams)"
    )
    parser.add_argument(
        "--output", default="thresholds.json",
        help="Output JSON file name (relative to EmotionModels/ or absolute). Default: thresholds.json"
    )
    parser.add_argument(
        "--n_thresholds", default=99, type=int,
        help="Number of threshold candidates to try per class (default: 99)"
    )
    args = parser.parse_args()

    optimize_thresholds(
        checkpoint_path=args.checkpoint,
        output_json=args.output,
        dev_csv=args.dev_csv,
        n_thresholds=args.n_thresholds,
    )
