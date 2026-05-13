"""
test.py — Evaluate a trained GraphBiasLabels checkpoint
=========================================================
Usage:
    python test.py --checkpoint checkpoints/best-epoch=042-val_f1_macro=0.7812.ckpt
    python test.py --checkpoint checkpoints/last.ckpt --graph_path graph.pt
    python test.py --checkpoint checkpoints/last.ckpt --split all      # train+val+test
    python test.py --checkpoint checkpoints/last.ckpt --leakage_check  # isolation test

Prints a full sklearn classification_report for the requested split.
--leakage_check additionally runs an isolation evaluation: all edges that
touch training nodes are removed before inference, revealing whether test
accuracy was inflated by label leakage through graph edges.
"""

import argparse
import logging
import os

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report
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
    parser.add_argument("--leakage_check", action="store_true",
                        help="Also run the isolation test: strips all edges "
                             "touching training nodes and re-evaluates on test "
                             "split to detect label leakage through graph edges")
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices",     type=int, default=1)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Leakage / isolation check
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_isolated(model, graph, device):
    """
    Re-runs inference after removing every edge that touches a training node.

    Why this matters
    ----------------
    In a transductive GNN each test node is IN the graph during training.
    If a test node has a direct edge to a training node, the GAT can aggregate
    that neighbor's feature vector (which correlates strongly with its known
    label) and effectively "copy" the label instead of learning from the text.

    Interpretation
    --------------
    - Accuracy stays similar  → model learned real signal from embeddings ✅
    - Accuracy drops sharply  → model was leaning on labeled neighbors   ⚠️
    """
    model.eval()
    graph = graph.to(device)

    train_indices = graph.train_mask.nonzero(as_tuple=True)[0]
    train_set     = set(train_indices.tolist())

    # Keep only edges where NEITHER endpoint is a training node
    src, dst = graph.edge_index
    keep_mask = torch.tensor(
        [s.item() not in train_set and d.item() not in train_set
         for s, d in zip(src, dst)],
        dtype=torch.bool, device=device,
    )

    isolated_edge_index = graph.edge_index[:, keep_mask]

    edges_before = graph.edge_index.shape[1]
    edges_after  = isolated_edge_index.shape[1]
    log.info(
        f"  Edges before isolation : {edges_before:,}\n"
        f"  Edges after  isolation : {edges_after:,} "
        f"({(edges_before - edges_after):,} removed)"
    )

    logits = model(graph.x, graph.emotions, isolated_edge_index)
    preds  = logits.argmax(dim=-1)

    valid  = graph.test_mask & (graph.y != -1)
    y_true = graph.y[valid].cpu().numpy()
    y_pred = preds[valid].cpu().numpy()

    acc = float((y_true == y_pred).mean())
    f1  = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    report = classification_report(
        y_true, y_pred,
        target_names=["Non-biased", "Biased"],
        zero_division=0,
    )

    log.info("\n" + "=" * 60)
    log.info("  ISOLATED TEST EVALUATION (edges to train nodes removed)")
    log.info("=" * 60)
    log.info(f"  Accuracy : {acc:.4f}")
    log.info(f"  F1 Macro : {f1:.4f}")
    log.info(f"\n{report}")

    return acc, f1


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

    # ------------------------------------------------------------------
    # 5. Optional leakage / isolation check
    # ------------------------------------------------------------------
    if args.leakage_check:
        log.info("\nRunning leakage isolation check …")
        acc_full = None
        if "test" in splits_to_run:
            results_test = model._evaluate(graph.to(device), graph.test_mask.to(device))
            acc_full = results_test["accuracy"]
            f1_full  = results_test["f1_macro"]

        acc_iso, f1_iso = evaluate_isolated(model, graph, device)

        if acc_full is not None:
            acc_delta = acc_iso - acc_full
            f1_delta  = f1_iso  - f1_full
            log.info("\n--- Leakage delta (isolated − full) ---")
            log.info(f"  Δ Accuracy : {acc_delta:+.4f}")
            log.info(f"  Δ F1 Macro : {f1_delta:+.4f}")
            if abs(acc_delta) < 0.03:
                log.info("  ✅ Minimal drop — model appears to use real text features")
            else:
                log.info("  ⚠️  Notable drop — model may be leaking labels through edges")


if __name__ == "__main__":
    main()
