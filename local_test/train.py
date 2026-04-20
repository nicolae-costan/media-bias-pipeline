#!/usr/bin/env python3
"""
Module A: Sentence-Level Bias & Emotion Engine — CLI Entry Point
================================================================
Usage:
    # Quick smoke test on sample data (CPU)
    python local_test/train.py \\
        --train_csv local_test/sample_train.csv \\
        --dev_csv   local_test/sample_dev.csv \\
        --test_csv  local_test/sample_test.csv \\
        --max_epochs_phase1 1 --max_epochs_phase2 1 \\
        --batch_size 4 --device cpu

    # Full GPU training on the real dataset
    python local_test/train.py \\
        --train_csv Resources/UsVsThem_train_public.csv \\
        --dev_csv   Resources/UsVsThem_valid_public.csv \\
        --test_csv  Resources/UsVsThem_test_public.csv \\
        --max_epochs_phase1 8 --max_epochs_phase2 3 \\
        --batch_size 16 --device cuda
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# ── Ensure local_test/ is on the import path ────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import RoBERTaMTL
from dataset import build_dataloaders
from trainer import BiasTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Module A: RoBERTa Multi-Task Bias & Emotion Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ────────────────────────────────────────────────────────────
    p.add_argument("--train_csv", type=str, required=True,
                   help="Path to training CSV")
    p.add_argument("--dev_csv", type=str, required=True,
                   help="Path to validation CSV")
    p.add_argument("--test_csv", type=str, required=True,
                   help="Path to test CSV")

    # ── Model ───────────────────────────────────────────────────────────
    p.add_argument("--encoder_model", type=str, default="roberta-base",
                   help="HuggingFace model identifier")
    p.add_argument("--extra_dropout", type=float, default=0.1,
                   help="Additional dropout on top of model defaults")
    p.add_argument("--max_length", type=int, default=512,
                   help="Max sequence length for tokenization")

    # ── Loss Weights ────────────────────────────────────────────────────
    p.add_argument("--weight_bias", type=float, default=1.0,
                   help="Loss weight for bias regression head")
    p.add_argument("--weight_emotion", type=float, default=1.0,
                   help="Loss weight for emotion multi-label head")
    p.add_argument("--weight_social", type=float, default=1.0,
                   help="Loss weight for social group head")

    # ── Training ────────────────────────────────────────────────────────
    p.add_argument("--batch_size", type=int, default=16,
                   help="Batch size for train/val/test")
    p.add_argument("--learning_rate", type=float, default=2e-5,
                   help="Base learning rate for the shared encoder")
    p.add_argument("--head_lr_multiplier", type=float, default=10.0,
                   help="LR multiplier for task-specific heads")
    p.add_argument("--warmup_proportion", type=float, default=0.1,
                   help="Fraction of total steps for LR warmup")
    p.add_argument("--max_epochs_phase1", type=int, default=8,
                   help="Max epochs for Phase 1 (full MTL)")
    p.add_argument("--max_epochs_phase2", type=int, default=3,
                   help="Max epochs for Phase 2 (social group ablated)")
    p.add_argument("--patience", type=int, default=5,
                   help="Early stopping patience (epochs)")
    p.add_argument("--accumulate_grad_batches", type=int, default=1,
                   help="Gradient accumulation steps (effective batch = batch_size * this)")

    # ── Hardware ────────────────────────────────────────────────────────
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cuda", "cpu"],
                   help="Device to train on")
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader worker processes")

    # ── Output ──────────────────────────────────────────────────────────
    p.add_argument("--output_dir", type=str, default="outputs/module_a",
                   help="Directory for checkpoints, logs, predictions")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Logging ─────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("train")

    # ── Seed ────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Device ──────────────────────────────────────────────────────────
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    log.info(f"Device: {device}")
    if device == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.set_float32_matmul_precision("high")

    # ── Data ────────────────────────────────────────────────────────────
    log.info("Loading datasets...")
    train_loader, val_loader, test_loader, group_encoder, emotion_pos_weights = build_dataloaders(
        train_csv=args.train_csv,
        dev_csv=args.dev_csv,
        test_csv=args.test_csv,
        model_name=args.encoder_model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
    )

    num_groups = len(group_encoder.classes_)
    log.info(f"Datasets loaded — {num_groups} social groups detected")

    # ── Model ───────────────────────────────────────────────────────────
    log.info("Initializing RoBERTaMTL model...")
    model = RoBERTaMTL(
        model_name=args.encoder_model,
        num_social_groups=num_groups,
        extra_dropout=args.extra_dropout,
        loss_weights={
            "bias": args.weight_bias,
            "emotion": args.weight_emotion,
            "social": args.weight_social,
        },
        emotion_pos_weights=emotion_pos_weights,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model parameters: {total_params:,} total, {trainable:,} trainable")

    # ── Trainer ─────────────────────────────────────────────────────────
    from dataset import EMOTION_COLUMNS

    trainer = BiasTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=args.learning_rate,
        head_lr_multiplier=args.head_lr_multiplier,
        max_epochs_phase1=args.max_epochs_phase1,
        max_epochs_phase2=args.max_epochs_phase2,
        warmup_proportion=args.warmup_proportion,
        output_dir=args.output_dir,
        patience=args.patience,
        emotion_columns=EMOTION_COLUMNS,
        group_classes=list(group_encoder.classes_),
        accumulate_grad_batches=args.accumulate_grad_batches,
    )

    # ── Run ──────────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("  MODULE A: Sentence-Level Bias & Emotion Engine")
    log.info(f"  Phase 1: {args.max_epochs_phase1} epochs (full MTL)")
    log.info(f"  Phase 2: {args.max_epochs_phase2} epochs (social ablated)")
    log.info(f"  Batch size: {args.batch_size} | LR: {args.learning_rate}")
    log.info(f"  Output: {args.output_dir}")
    log.info("=" * 70)
    log.info("")

    trainer.run()

    log.info("Done!")


if __name__ == "__main__":
    main()
