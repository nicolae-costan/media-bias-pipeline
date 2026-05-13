"""
test.py — Evaluate a trained GraphBiasLabels checkpoint
=========================================================
Usage:
    python test.py --checkpoint checkpoints/best-epoch=042-val_f1_macro=0.7812.ckpt
    python test.py --checkpoint checkpoints/last.ckpt --graph_path graph.pt
    python test.py --checkpoint checkpoints/last.ckpt --split all   # train+val+test

Prints a full sklearn classification_report for the requested split and logs
key metrics to TensorBoard (if --log_dir is given).
"""

import argparse
import logging
import os

import torch
import pytorch_lightning as pl
from torch_geometric.data import Data

from GraphModel import GraphBiasLabels

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tiny single-graph dataset helper (same as train.py)
# ---------------------------------------------------------------------------

class SingleGraphDataset(torch.utils.data.Dataset):
    def __init__(self, data: Data):
        self.data = data

    def __len__(self):
        return 1

    def __getitem__(self, _):
        return self.data


def _identity_collate(batch):
    return batch[0]


def make_dataloader(data: Data) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        SingleGraphDataset(data),
        batch_size=1,
        collate_fn=_identity_collate,
        num_workers=0,
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained GraphBiasLabels checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the .ckpt file to evaluate")
    parser.add_argument("--graph_path", type=str, default="graph.pt",
                        help="Path to the graph .pt file")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test", "all"],
                        help="Which mask(s) to evaluate on")
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices",     type=int, default=1)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    # ------------------------------------------------------------------
    # 1. Load graph
    # ------------------------------------------------------------------
    if not os.path.exists(args.graph_path):
        raise FileNotFoundError(
            f"Graph file not found: {args.graph_path}. "
            "Run build_graph.py first."
        )
    log.info(f"Loading graph from '{args.graph_path}' …")
    graph: Data = torch.load(args.graph_path, weights_only=False)
    log.info(
        f"Graph — nodes: {graph.num_nodes:,} | edges: {graph.num_edges:,}"
    )

    # ------------------------------------------------------------------
    # 2. Load model from checkpoint
    # ------------------------------------------------------------------
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    log.info(f"Loading model from '{args.checkpoint}' …")
    model = GraphBiasLabels.load_from_checkpoint(args.checkpoint)
    model.graph_data   = graph
    model.train_data   = graph.train_mask
    model.val_data     = graph.val_mask
    model.predict_data = graph.test_mask
    model.eval()

    # ------------------------------------------------------------------
    # 3. Choose which splits to evaluate
    # ------------------------------------------------------------------
    splits_to_run = (
        ["train", "val", "test"] if args.split == "all" else [args.split]
    )

    mask_map = {
        "train": graph.train_mask,
        "val":   graph.val_mask,
        "test":  graph.test_mask,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # ------------------------------------------------------------------
    # 4. Evaluate each split
    # ------------------------------------------------------------------
    for split_name in splits_to_run:
        mask = mask_map[split_name]
        n_nodes = int(mask.sum().item())
        log.info(f"\n{'='*60}")
        log.info(f"  Evaluating on '{split_name}' split  ({n_nodes:,} nodes)")
        log.info(f"{'='*60}")

        results = model._evaluate(graph.to(device), mask.to(device))

        log.info(f"  Loss     : {results['loss']:.4f}")
        log.info(f"  Accuracy : {results['accuracy']:.4f}")
        log.info(f"  F1 Macro : {results['f1_macro']:.4f}")
        log.info(f"  F1 Biased: {results['f1_biased']:.4f}")
        log.info(f"\n{results['report']}")

    log.info("Evaluation complete.")


if __name__ == "__main__":
    main()
