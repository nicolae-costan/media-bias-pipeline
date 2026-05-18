import argparse
import os
import sys

import pandas as pd
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GRAPH_DIR = os.path.join(ROOT, "GraphNeuralNetwork")
if GRAPH_DIR not in sys.path:
    sys.path.insert(0, GRAPH_DIR)

from GraphModel import GraphBiasLabels


def _read_babe_ids(path: str) -> set[str]:
    if not path or not os.path.exists(path):
        return set()
    df = pd.read_csv(path, sep=";", on_bad_lines="skip")
    return set(df["article_id"].dropna().astype(str))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Export high-confidence graph pseudo-labels for non-BABE articles.")
    parser.add_argument("--graph_path", default="GraphNeuralNetwork/graph.pt")
    parser.add_argument("--checkpoint", default="GraphNeuralNetwork/checkpoints/last.ckpt")
    parser.add_argument("--babe_csv", default="data/final_labels_MBIC.csv")
    parser.add_argument("--output", default="data/bias_transformer/graph_pseudo_labels.csv")
    parser.add_argument("--min_confidence", type=float, default=0.90)
    args = parser.parse_args()

    if not os.path.exists(args.graph_path):
        raise FileNotFoundError(f"Graph file not found: {args.graph_path}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    graph = torch.load(args.graph_path, weights_only=False)
    model = GraphBiasLabels.load_from_checkpoint(args.checkpoint, strict=False)
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

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    pd.DataFrame(records).to_csv(args.output, index=False)
    print(f"[graph_pseudo_labels] Saved {len(records)} rows to {args.output}")


if __name__ == "__main__":
    main()
