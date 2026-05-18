"""
random_forest_baseline.py
=========================
A non-graph baseline so we can tell whether the GAT is actually adding
anything over a plain tabular classifier on the same features and splits.

Inputs come from graph.pt — exactly the same CLS embeddings, emotion scores,
labels, and train/val/test masks the GNN sees. This makes the comparison
apples-to-apples: any gap between RF and the GAT is attributable to the
graph/model, not to data prep differences.

Usage:
    python random_forest_baseline.py
    python random_forest_baseline.py --graph_path graph.pt --n_estimators 1000
    python random_forest_baseline.py --features both         # CLS + emotions (default)
    python random_forest_baseline.py --features cls          # CLS embeddings only
    python random_forest_baseline.py --features emotions     # emotions only
"""

import argparse
import os

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def get_args():
    p = argparse.ArgumentParser(
        description="Random Forest baseline on the GNN's graph.pt features/splits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--graph_path", type=str, default="graph.pt",
                   help="Path to the .pt file produced by build_graph.py")
    p.add_argument("--features", choices=["both", "cls", "emotions"], default="both",
                   help="Which features to feed the RF")
    p.add_argument("--n_estimators", type=int, default=500)
    p.add_argument("--max_depth", type=int, default=None,
                   help="None = grow trees fully")
    p.add_argument("--min_samples_leaf", type=int, default=2)
    p.add_argument("--class_weight", type=str, default="balanced",
                   choices=["balanced", "balanced_subsample", "none"])
    p.add_argument("--use_sample_weights", action="store_true",
                   help="Use graph.label_weights (annotator agreement) as sample weights")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_feature_matrix(graph, mode: str) -> np.ndarray:
    cls = graph.x.numpy()
    emo = graph.emotions.numpy()
    if mode == "cls":
        return cls
    if mode == "emotions":
        return emo
    return np.concatenate([cls, emo], axis=1)


def report(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if y_true.size == 0:
        print(f"\n=== {name} === (empty mask, skipped)")
        return
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_biased = f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    print(f"\n=== {name} (n={len(y_true)}) ===")
    print(f"  accuracy : {acc:.4f}")
    print(f"  f1_macro : {f1_macro:.4f}")
    print(f"  f1_biased: {f1_biased:.4f}")
    print("  confusion matrix [rows=true, cols=pred]:")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"             pred_Non  pred_Biased")
    print(f"   true_Non    {cm[0,0]:>6d}      {cm[0,1]:>6d}")
    print(f"   true_Bias   {cm[1,0]:>6d}      {cm[1,1]:>6d}")
    print(classification_report(
        y_true, y_pred,
        target_names=["Non-biased", "Biased"],
        zero_division=0,
        digits=4,
    ))


def main():
    args = get_args()

    if not os.path.exists(args.graph_path):
        raise FileNotFoundError(
            f"{args.graph_path} not found — run build_graph.py first."
        )

    print(f"Loading graph from {args.graph_path} …")
    graph = torch.load(args.graph_path, weights_only=False)

    # Pull tensors → numpy for sklearn
    y = graph.y.numpy()
    train_mask = graph.train_mask.numpy().astype(bool)
    val_mask = graph.val_mask.numpy().astype(bool)
    test_mask = graph.test_mask.numpy().astype(bool)
    sample_weights = (
        graph.label_weights.numpy()
        if hasattr(graph, "label_weights") else None
    )

    X = build_feature_matrix(graph, args.features)
    print(f"Feature matrix shape: {X.shape}  (mode={args.features})")
    print(f"  train labeled: {int((train_mask & (y != -1)).sum()):,}")
    print(f"  val   labeled: {int((val_mask   & (y != -1)).sum()):,}")
    print(f"  test  labeled: {int((test_mask  & (y != -1)).sum()):,}")

    # Restrict each split to labeled rows
    train_idx = np.where(train_mask & (y != -1))[0]
    val_idx   = np.where(val_mask   & (y != -1))[0]
    test_idx  = np.where(test_mask  & (y != -1))[0]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    # Class balance summary on the train set
    counts = np.bincount(y_train, minlength=2)
    print(f"  train class balance — Non-biased: {counts[0]} | Biased: {counts[1]} "
          f"(majority-baseline acc = {counts.max() / counts.sum():.4f})")

    class_weight = None if args.class_weight == "none" else args.class_weight
    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=args.seed,
    )

    fit_kwargs = {}
    if args.use_sample_weights and sample_weights is not None:
        fit_kwargs["sample_weight"] = sample_weights[train_idx]
        print("  Using annotator agreement as sample_weight for training.")

    print(f"\nTraining RandomForest "
          f"(n_estimators={args.n_estimators}, max_depth={args.max_depth}, "
          f"class_weight={args.class_weight}) …")
    rf.fit(X_train, y_train, **fit_kwargs)

    # Predictions
    y_train_pred = rf.predict(X_train)
    y_val_pred   = rf.predict(X_val)
    y_test_pred  = rf.predict(X_test)

    report("TRAIN", y_train, y_train_pred)
    report("VAL",   y_val,   y_val_pred)
    report("TEST",  y_test,  y_test_pred)

    # Bonus: top feature importances (only meaningful when emotions are included
    # since the CLS dims are uninterpretable).
    if args.features in ("both", "emotions"):
        emotion_names = [
            "anger", "disgust", "fear", "joy",
            "optimism", "sadness", "neutral",
        ]
        if args.features == "emotions":
            importances = rf.feature_importances_
        else:
            importances = rf.feature_importances_[-len(emotion_names):]
        order = np.argsort(importances)[::-1]
        print("\nTop emotion feature importances:")
        for rank, j in enumerate(order, 1):
            print(f"  {rank}. {emotion_names[j]:<10s} {importances[j]:.4f}")


if __name__ == "__main__":
    main()
