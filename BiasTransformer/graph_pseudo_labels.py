import argparse
import os
import sys
from argparse import Namespace

import pandas as pd
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GRAPH_DIR = os.path.join(ROOT, "GraphNeuralNetwork")
if GRAPH_DIR not in sys.path:
    sys.path.insert(0, GRAPH_DIR)

from GraphModel import GraphBiasLabels


def _load_graph_model(checkpoint_path: str, graph, graph_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hparams = dict(checkpoint.get("hyper_parameters", {}))
    hparams["cls_input_dim"] = graph.x.shape[1]
    hparams["emo_input_dim"] = graph.emotions.shape[1]

    hparams["graph_path"] = graph_path

    model = GraphBiasLabels(Namespace(**hparams))
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    return model


def _read_babe_ids(path: str) -> set[str]:
    if not path or not os.path.exists(path):
        return set()
    df = pd.read_csv(path, sep=";", on_bad_lines="skip")
    return set(df["article_id"].dropna().astype(str))


def _cap_pseudo_labels(records: list[dict], max_per_class: int | None, balanced_per_class: bool) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return df

    df = df.sort_values("confidence", ascending=False).reset_index(drop=True)
    before_counts = df["predicted_label"].value_counts()
    print("[graph_pseudo_labels] Counts before class capping:")
    print(before_counts.to_string())

    if max_per_class is None:
        return df

    if max_per_class <= 0:
        raise ValueError("--max_per_class must be greater than 0 when provided")

    if balanced_per_class:
        counts = df["predicted_label"].value_counts()
        if len(counts) < 2:
            print("[graph_pseudo_labels] Warning: only one class available after filtering; balanced output is empty.")
            return df.iloc[0:0]
        keep_per_class = min(max_per_class, int(counts.min()))
        capped = (
            df.groupby("predicted_label", group_keys=False)
            .head(keep_per_class)
            .sort_values("confidence", ascending=False)
            .reset_index(drop=True)
        )
        print(f"[graph_pseudo_labels] Balanced cap: keeping {keep_per_class} rows per class.")
        return capped

    capped = (
        df.groupby("predicted_label", group_keys=False)
        .head(max_per_class)
        .sort_values("confidence", ascending=False)
        .reset_index(drop=True)
    )
    print(f"[graph_pseudo_labels] Per-class cap: keeping up to {max_per_class} rows per class.")
    return capped


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Export high-confidence graph pseudo-labels for non-BABE articles.")
    parser.add_argument("--graph_path", default="GraphNeuralNetwork/graph.pt")
    parser.add_argument("--checkpoint", default="GraphNeuralNetwork/checkpoints/last.ckpt")
    parser.add_argument("--babe_csv", default="data/final_labels_MBIC.csv")
    parser.add_argument("--output", default="data/bias_transformer/graph_pseudo_labels.csv")
    parser.add_argument("--min_confidence", type=float, default=0.90)
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=None,
        help="Optional cap on pseudo-label rows per predicted class after confidence filtering.",
    )
    parser.add_argument(
        "--balanced_per_class",
        action="store_true",
        help="Keep the same number of rows for each predicted class, capped by --max_per_class.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.graph_path):
        raise FileNotFoundError(f"Graph file not found: {args.graph_path}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    graph = torch.load(args.graph_path, weights_only=False)
    model = _load_graph_model(args.checkpoint, graph, args.graph_path)
    model.eval()

    logits = model(graph.x, graph.emotions, graph.edge_index)
    probs = torch.softmax(logits, dim=-1)
    confidence, preds = probs.max(dim=-1)
    prob_biased = probs[:, 1]

    babe_ids = _read_babe_ids(args.babe_csv)
    label_map = {0: "Non-biased", 1: "Biased"}
    records = []
    for idx, article_id in enumerate(graph.article_ids):
        article_id = str(article_id)
        if article_id in babe_ids:
            continue
        if graph.y[idx].item() != -1:
            continue
        conf = float(confidence[idx].item())
        if conf < args.min_confidence:
            continue
        records.append(
            {
                "article_id": article_id,
                "predicted_label": label_map[int(preds[idx].item())],
                "prob_biased": float(prob_biased[idx].item()),
                "confidence": conf,
            }
        )

    df = _cap_pseudo_labels(records, args.max_per_class, args.balanced_per_class)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print("[graph_pseudo_labels] Counts saved:")
    if df.empty:
        print("No rows")
    else:
        print(df["predicted_label"].value_counts().to_string())
        print("[graph_pseudo_labels] Confidence summary:")
        print(df["confidence"].describe().to_string())
    print(f"[graph_pseudo_labels] Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
