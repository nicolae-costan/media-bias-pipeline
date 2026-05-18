"""
train.py — Training entry-point for GraphBiasLabels
=====================================================
Usage:
    python train.py                          # use all defaults from GraphModel.add_model_specific_args()
    python train.py --max_epochs 100 --lr_gat 1e-3 --graph_path graph.pt
    python train.py --accelerator gpu --devices 1

The graph is a single, whole-graph object (transductive setting), so the
DataLoader returns the graph itself as the "batch" every step.  We wrap it in
a tiny list-dataset so PL can iterate over it.
"""

import argparse
import logging
import os
import warnings

import torch
from torch_geometric.data import Data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger

from GraphModel import GraphBiasLabels

# Silence PL's `num_workers` recommendation: this is a transductive single-graph
# setup with exactly one "batch" per epoch (the whole graph), so multi-worker
# loading would not help and would in fact add IPC overhead. We match on the
# message text only — the warning category has moved between PL versions.
warnings.filterwarnings("ignore", message=".*does not have many workers.*")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tiny single-graph dataset / dataloader
# ---------------------------------------------------------------------------

class SingleGraphDataset(torch.utils.data.Dataset):
    """Wraps a single PyG Data object so PL's DataLoader can iterate over it."""
    def __init__(self, data: Data):
        self.data = data

    def __len__(self):
        return 1

    def __getitem__(self, _):
        return self.data


def _identity_collate(batch):
    """Return the single graph object unchanged (no stacking needed)."""
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
        description="Train the GraphBiasLabels GNN model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Training / PL Trainer args ---
    parser.add_argument("--max_epochs",      type=int,   default=150)
    parser.add_argument("--accelerator",     type=str,   default="auto",
                        help="'cpu', 'gpu', 'mps', or 'auto'")
    parser.add_argument("--devices",         type=int,   default=1)
    parser.add_argument("--log_dir",         type=str,   default="tb_logs",
                        help="TensorBoard log root directory")
    parser.add_argument("--experiment_name", type=str,   default="GraphBiasLabels")
    parser.add_argument("--ckpt_dir",        type=str,   default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--early_stop_patience", type=int, default=20,
                        help="EarlyStopping patience (epochs); 0 to disable")
    parser.add_argument("--resume_ckpt",     type=str,   default=None,
                        help="Path to a checkpoint to resume training from")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for torch / numpy / python — set for reproducible runs")
    parser.add_argument("--grad_clip_val", type=float, default=1.0,
                        help="Trainer gradient_clip_val (0 to disable)")

    # Let the model declare its own hyperparameters
    parser = GraphBiasLabels.add_model_specific_args(parser)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    # ------------------------------------------------------------------
    # 0. Reproducibility + Tensor-Cores fast path
    # ------------------------------------------------------------------
    # Seed everything (python, numpy, torch, cuda) so two runs with the same
    # args produce comparable numbers. Without this you can't tell whether a
    # code change moved the metric or you just got a lucky init.
    pl.seed_everything(args.seed, workers=True)

    # Silence the Tensor Cores warning + actually use them. 'high' keeps
    # float32 matmul accurate enough for our scale while picking up the
    # speed of TF32 on Ampere/Ada GPUs.
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    # ------------------------------------------------------------------
    # 1. Load graph
    # ------------------------------------------------------------------
    if not os.path.exists(args.graph_path):
        log.error(f"Graph file not found: {args.graph_path}. Run build_graph.py first.")
        raise FileNotFoundError(args.graph_path)

    log.info(f"Loading graph from '{args.graph_path}' …")
    graph: Data = torch.load(args.graph_path, weights_only=False)

    log.info(
        f"Graph loaded — nodes: {graph.num_nodes:,} | edges: {graph.num_edges:,} | "
        f"train: {graph.train_mask.sum():,} | val: {graph.val_mask.sum():,} | "
        f"test: {graph.test_mask.sum():,}"
    )

    # Class-balance sanity check on the train split — a heavily skewed split
    # would make 'accuracy' meaningless and explain a stuck-looking model.
    for split_name, mask in (("train", graph.train_mask),
                             ("val",   graph.val_mask),
                             ("test",  graph.test_mask)):
        y = graph.y[mask & (graph.y != -1)]
        if y.numel() == 0:
            continue
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        majority = max(n_pos, n_neg) / max(n_pos + n_neg, 1)
        log.info(
            f"{split_name:>5s} balance — Non-biased: {n_neg:>5d} | "
            f"Biased: {n_pos:>5d} | majority-baseline acc: {majority:.3f}"
        )

    # ------------------------------------------------------------------
    # 2. Inject input dims from graph (avoid shape mismatches)
    # ------------------------------------------------------------------
    args.cls_input_dim = graph.x.shape[1]          # 768 CLS embeddings
    args.emo_input_dim = graph.emotions.shape[1]   # 13 emotion scores
    log.info(f"Input dims — CLS: {args.cls_input_dim}, Emotions: {args.emo_input_dim}")

    # ------------------------------------------------------------------
    # 3. Build model
    # ------------------------------------------------------------------
    model = GraphBiasLabels(hparams=args)

    # Override graph_data loaded inside build_model() with the already-
    # loaded object so we don't hit disk a second time.
    model.graph_data   = graph
    model.train_data   = graph.train_mask
    model.val_data     = graph.val_mask
    model.predict_data = graph.test_mask

    # ------------------------------------------------------------------
    # 4. DataLoaders (whole-graph, transductive)
    # ------------------------------------------------------------------
    train_loader = make_dataloader(graph)
    val_loader   = make_dataloader(graph)

    # ------------------------------------------------------------------
    # 5. Callbacks
    # ------------------------------------------------------------------
    os.makedirs(args.ckpt_dir, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=args.ckpt_dir,
            filename="best-{epoch:03d}-{val_f1_macro:.4f}",
            monitor="val_f1_macro",
            mode="max",
            save_top_k=3,
            save_last=True,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    if args.early_stop_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val_f1_macro",
                patience=args.early_stop_patience,
                mode="max",
                verbose=True,
            )
        )

    # ------------------------------------------------------------------
    # 6. Logger
    # ------------------------------------------------------------------
    tb_logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.experiment_name,
    )

    # ------------------------------------------------------------------
    # 7. Trainer
    # ------------------------------------------------------------------
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=callbacks,
        logger=tb_logger,
        log_every_n_steps=1,
        # Gradient clipping: cheap insurance against GAT attention blow-ups
        # on a small graph; set --grad_clip_val 0 to disable.
        gradient_clip_val=args.grad_clip_val if args.grad_clip_val > 0 else None,
        # cudnn benchmark mode for fixed-shape graphs — we always pass the same
        # tensor shapes through the model, so let cudnn pick the fastest kernels.
        benchmark=True,
        deterministic=False,  # set True for strict reproducibility at a speed cost
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # ------------------------------------------------------------------
    # 8. Fit
    # ------------------------------------------------------------------
    log.info("Starting training …")
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_ckpt)

    # ------------------------------------------------------------------
    # 9. Run test split after training
    # ------------------------------------------------------------------
    log.info("Running evaluation on test split …")
    test_loader = make_dataloader(graph)
    trainer.test(model, test_loader, ckpt_path="best")

    log.info("Training complete.")


if __name__ == "__main__":
    main()
