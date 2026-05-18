import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from dataloader import LABEL_TO_ID


def _read_babe_ids(path: str) -> set[str]:
    if not path or not os.path.exists(path):
        return set()
    df = pd.read_csv(path, sep=";", on_bad_lines="skip")
    return set(df["article_id"].dropna().astype(str))


def _normalize_body(text: str) -> str:
    return " ".join(str(text).split()).lower()


def _clean_labeled_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["label"].isin(LABEL_TO_ID)].copy()
    df = df.dropna(subset=["article_id", "body", "label"])
    df["article_id"] = df["article_id"].astype(str)
    df["body_norm"] = df["body"].map(_normalize_body)
    df = df.drop_duplicates(subset=["article_id"])
    df = df.drop_duplicates(subset=["body_norm"])
    return df.drop(columns=["body_norm"]).reset_index(drop=True)


def _stratified_split(df: pd.DataFrame, seed: int, train_size: float, valid_size: float):
    train_df, tmp_df = train_test_split(
        df,
        train_size=train_size,
        random_state=seed,
        stratify=df["label"],
    )
    relative_valid = valid_size / (1.0 - train_size)
    valid_df, test_df = train_test_split(
        tmp_df,
        train_size=relative_valid,
        random_state=seed,
        stratify=tmp_df["label"],
    )
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_pseudo_stage(args, merged: pd.DataFrame, babe_ids: set[str], output_dir: Path):
    if not args.graph_predictions_csv:
        print("[prepare_data] No --graph_predictions_csv provided; skipping pseudo-label pretrain stage.")
        return

    pred = pd.read_csv(args.graph_predictions_csv)
    required = {"article_id", "predicted_label", "confidence"}
    missing = required - set(pred.columns)
    if missing:
        raise ValueError(f"Graph prediction CSV missing columns: {sorted(missing)}")

    pred["article_id"] = pred["article_id"].astype(str)
    pred = pred[pred["confidence"] >= args.pseudo_min_confidence].copy()
    pred = pred[pred["predicted_label"].isin(LABEL_TO_ID)]

    merged_small = merged[["article_id", "body"]].copy()
    merged_small["article_id"] = merged_small["article_id"].astype(str)
    stage = pred.merge(merged_small, on="article_id", how="inner")
    stage = stage[~stage["article_id"].isin(babe_ids)]
    stage = stage.rename(columns={"predicted_label": "label", "confidence": "sample_weight"})
    stage = _clean_labeled_frame(stage[["article_id", "body", "label", "sample_weight"]])

    if len(stage) == 0:
        raise ValueError("No pseudo-labeled non-BABE rows remained after filtering.")

    train_df, valid_df, test_df = _stratified_split(stage, args.seed, args.train_size, args.valid_size)
    train_df.to_csv(output_dir / "pretrain_train.csv", index=False)
    valid_df.to_csv(output_dir / "pretrain_valid.csv", index=False)
    test_df.to_csv(output_dir / "pretrain_test.csv", index=False)
    print(f"[prepare_data] Pseudo stage rows: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")


def build_finetune_stage(args, merged: pd.DataFrame, output_dir: Path):
    consensus = pd.read_csv(args.consensus_csv)
    consensus["article_id"] = consensus["article_id"].astype(str)
    consensus = consensus.rename(columns={"consensus_label": "label", "agreement": "sample_weight"})

    merged_small = merged[["article_id", "body"]].copy()
    merged_small["article_id"] = merged_small["article_id"].astype(str)
    stage = consensus.merge(merged_small, on="article_id", how="inner")
    stage = _clean_labeled_frame(stage[["article_id", "body", "label", "sample_weight"]])

    if len(stage) == 0:
        raise ValueError("No consensus-labeled rows remained after joining with merged data.")

    train_df, valid_df, test_df = _stratified_split(stage, args.seed, args.train_size, args.valid_size)
    train_df.to_csv(output_dir / "finetune_train.csv", index=False)
    valid_df.to_csv(output_dir / "finetune_valid.csv", index=False)
    test_df.to_csv(output_dir / "finetune_test.csv", index=False)
    print(f"[prepare_data] Fine-tune stage rows: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")


def main():
    parser = argparse.ArgumentParser(description="Prepare two-stage bias transformer datasets.")
    parser.add_argument("--merged_csv", default="data/merged_clean_data.csv")
    parser.add_argument("--consensus_csv", default="data/consensus_labels_sg1_sg2.csv")
    parser.add_argument("--babe_csv", default="data/final_labels_MBIC.csv")
    parser.add_argument("--graph_predictions_csv", default=None)
    parser.add_argument("--output_dir", default="data/bias_transformer")
    parser.add_argument("--pseudo_min_confidence", type=float, default=0.90)
    parser.add_argument("--train_size", type=float, default=0.70)
    parser.add_argument("--valid_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged = pd.read_csv(args.merged_csv)
    merged["article_id"] = merged["article_id"].astype(str)
    babe_ids = _read_babe_ids(args.babe_csv)

    build_pseudo_stage(args, merged, babe_ids, output_dir)
    build_finetune_stage(args, merged, output_dir)


if __name__ == "__main__":
    main()
