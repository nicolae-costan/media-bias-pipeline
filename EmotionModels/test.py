"""
Runs a trained model on the Test Set using a saved checkpoint.
Displays metrics in the same style as optimize_thresholds.py.
"""
import os
import sys
import json
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import jaccard_score, f1_score, classification_report
from torch.utils.data import DataLoader

# Resolve imports whether run from project root or EmotionModels/
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from model import EmotionModel
from dataloader import RedditDataset, sentiment_analysis_dataset, MyCollator
from pytorch_lightning import seed_everything
from test_tube import HyperOptArgumentParser


def main(hparams) -> None:
    seed_everything(hparams.seed)

    # ── 1. Find & load checkpoint ────────────────────────────────────────────
    model = None
    
    checkpoint_path = hparams.checkpoint_path
    if not os.path.isabs(checkpoint_path):
        # Try as-is first, then relative to the script's directory
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.normpath(os.path.join(_script_dir, checkpoint_path))

    if os.path.exists(hparams.checkpoint_path):
        for file in os.listdir(hparams.checkpoint_path):
            if file.endswith(".ckpt"):
                ckpt_full_path = os.path.join(hparams.checkpoint_path, file)
                print(f"\n{'='*60}")
                print(f"  Loading checkpoint: {ckpt_full_path}")
                print(f"{'='*60}\n")
                model = EmotionModel.load_from_checkpoint(ckpt_full_path, strict=False)

                # Threshold resolution: next to checkpoint → EmotionModels/ → sibling folder
                candidates = [
                    os.path.join(os.path.dirname(hparams.checkpoint_path), "thresholds.json"),
                    os.path.join(_script_dir, "thresholds.json"),
                    os.path.join(_script_dir, "..", "EmotionModels", "thresholds.json"),
                ]
                for t_path in candidates:
                    t_path = os.path.normpath(t_path)
                    if os.path.exists(t_path):
                        model.load_thresholds(t_path)
                        print(f"--- Loaded thresholds from: {t_path} ---")
                        break
                else:
                    print("--- No thresholds.json found, using default 0.5 ---")
                break

    if model is None:
        raise FileNotFoundError(f"Could not find a .ckpt file in {hparams.checkpoint_path}")

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    torch.set_float32_matmul_precision('high')

    emotions = RedditDataset.EMOTION_COLUMNS
    num_emotions = len(emotions)
    print(f"  Emotion classes ({num_emotions}): {emotions}\n")

    # ── 2. Collect probabilities on the test set ─────────────────────────────
    print("Step A — Running inference on test set …")

    # Force num_workers=0 — Windows multiprocessing deadlocks with tokenizers
    model.hparams.loader_workers = 0

    dataset  = sentiment_analysis_dataset(model.hparams, train=False, val=False, test=True)
    collator = MyCollator(model.hparams.encoder_model, model.hparams.max_length)
    loader   = DataLoader(
        dataset,
        batch_size  = model.hparams.batch_size,
        collate_fn  = collator,
        num_workers = 0,          # critical on Windows
        pin_memory  = False,
    )

    all_probs, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="  inference"):
            input_ids, attention_mask = model._safe_squeeze(inputs)
            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            logits = model.forward(input_ids, attention_mask)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_targets.append(targets['labels_aux'].numpy())

    probs   = np.concatenate(all_probs,   axis=0)   # [N, num_emotions]
    targets = np.concatenate(all_targets, axis=0)   # [N, num_emotions]

    # ── 3. Apply tuned thresholds ────────────────────────────────────────────
    thresholds = model.thresholds.cpu().numpy()      # shape [num_emotions]
    preds = np.stack(
        [(probs[:, i] > thresholds[i]).astype(float) for i in range(num_emotions)],
        axis=1,
    )

    # ── 4. Per-class metrics table ───────────────────────────────────────────
    print(f"\nStep B — Per-class metrics on test set …\n")

    per_class_jaccard = jaccard_score(targets, preds, average=None,    zero_division=0)
    per_class_f1      = f1_score(     targets, preds, average=None,    zero_division=0)
    global_jaccard    = jaccard_score(targets, preds, average="macro", zero_division=0)
    global_f1         = f1_score(     targets, preds, average="macro", zero_division=0)

    header = f"  {'Emotion':<14} {'Threshold':>9}  {'Jaccard':>8}  {'F1':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, name in enumerate(emotions):
        print(f"  {name:<14} {thresholds[i]:>9.2f}  {per_class_jaccard[i]:>8.4f}  {per_class_f1[i]:>8.4f}")
    print("  " + "-" * (len(header) - 2))
    print(f"  {'MACRO':<14} {'':>9}  {global_jaccard:>8.4f}  {global_f1:>8.4f}")

    # ── 5. Baseline vs tuned summary ─────────────────────────────────────────
    print(f"\nStep C — Baseline (0.5) vs tuned thresholds …\n")

    base_preds    = (probs > 0.5).astype(float)
    base_jaccard  = jaccard_score(targets, base_preds, average="macro", zero_division=0)
    base_f1       = f1_score(     targets, base_preds, average="macro", zero_division=0)

    print(f"  {'':20} {'Jaccard':>8}  {'Macro-F1':>9}")
    print(f"  {'Baseline (0.5)':20} {base_jaccard:>8.4f}  {base_f1:>9.4f}")
    print(f"  {'Tuned':20} {global_jaccard:>8.4f}  {global_f1:>9.4f}")
    print(f"  {'Improvement':20} {global_jaccard - base_jaccard:>+8.4f}  {global_f1 - base_f1:>+9.4f}")

    # ── 6. Full classification report ────────────────────────────────────────
    print(f"\nStep D — Full classification report (tuned thresholds):\n")
    print(classification_report(
        targets, preds,
        target_names=emotions,
        zero_division=0,
        digits=4,
    ))


if __name__ == "__main__":
    parser = HyperOptArgumentParser(
        strategy="random_search",
        description="BERT Multi-task Tester",
        add_help=True,
    )

    parser.add_argument(
        "--checkpoint_path",
        default="tb_logs/emotion_classification/version_3/checkpoints",
        type=str,
        help="Path to the FOLDER containing the .ckpt file",
    )
    parser.add_argument("--seed",       type=int, default=3)
    parser.add_argument("--gpus",       type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=6)

    parser = EmotionModel.add_model_specific_args(parser)
    hparams = parser.parse_args()

    main(hparams)