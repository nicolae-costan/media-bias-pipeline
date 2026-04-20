"""
Two-Phase Training Loop with Ablation
======================================
Phase 1 — Full MTL:   All 3 heads active (bias + emotions + social group).
Phase 2 — Ablation:   Social group head frozen & zeroed; fine-tune on
                       bias + emotions only.

This replaces the PyTorch Lightning + test_tube pattern used in
SentimentClassification/ with a clean, explicit, pure-PyTorch loop.
"""

import csv
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import jaccard_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from model import RoBERTaMTL

log = logging.getLogger(__name__)


# ── Metrics ─────────────────────────────────────────────────────────────

def pearson_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute Pearson r between two 1-D tensors (stays on device)."""
    x = x.float().flatten()
    y = y.float().flatten()
    x_c = x - x.mean()
    y_c = y - y.mean()
    r = (x_c * y_c).sum() / (x_c.norm() * y_c.norm() + 1e-8)
    return r.item()


def emotion_jaccard(
    y_true: torch.Tensor, y_pred_logits: torch.Tensor
) -> float:
    """Macro-averaged Jaccard score for multi-label emotion predictions."""
    y_pred = (torch.sigmoid(y_pred_logits) > 0.5).cpu().numpy()
    y_true_np = y_true.cpu().numpy()
    return float(jaccard_score(y_true_np, y_pred, average="macro"))


def social_accuracy(
    y_true: torch.Tensor, y_pred_logits: torch.Tensor
) -> float:
    """Top-1 accuracy for social group classification."""
    preds = torch.argmax(y_pred_logits, dim=1)
    return (preds == y_true).float().mean().item()


# ── Trainer ─────────────────────────────────────────────────────────────

class BiasTrainer:
    """
    Explicit two-phase training loop for RoBERTaMTL.

    Args:
        model:              The RoBERTaMTL model instance.
        train_loader:       DataLoader for training data.
        val_loader:         DataLoader for validation data.
        test_loader:        DataLoader for test data.
        device:             "cuda" or "cpu".
        learning_rate:      Base learning rate for the shared encoder.
        head_lr_multiplier: Multiplier for task-head learning rates.
        max_epochs_phase1:  Epochs for full MTL training.
        max_epochs_phase2:  Epochs for ablation fine-tuning.
        warmup_proportion:  Fraction of total steps used for LR warmup.
        output_dir:         Directory for checkpoints, logs, predictions.
        patience:           Early stopping patience (epochs without improvement).
    """

    def __init__(
        self,
        model: RoBERTaMTL,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: str = "cuda",
        learning_rate: float = 2e-5,
        head_lr_multiplier: float = 10.0,
        max_epochs_phase1: int = 8,
        max_epochs_phase2: int = 3,
        warmup_proportion: float = 0.1,
        output_dir: str = "outputs",
        patience: int = 5,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = torch.device(device)
        self.lr = learning_rate
        self.head_lr_mul = head_lr_multiplier
        self.max_epochs_p1 = max_epochs_phase1
        self.max_epochs_p2 = max_epochs_phase2
        self.warmup_proportion = warmup_proportion
        self.output_dir = Path(output_dir)
        self.patience = patience

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "tb_logs"))
        self.best_val_loss = float("inf")
        self.global_step = 0

    # ── Optimizer & Scheduler Factory ───────────────────────────────────

    def _build_optimizer_and_scheduler(
        self, total_steps: int
    ) -> Tuple[AdamW, torch.optim.lr_scheduler.LambdaLR]:
        """
        Build AdamW with differential learning rates:
          - Shared encoder params at `lr`
          - Active head params at `lr * head_lr_multiplier`
        """
        param_groups = [
            {
                "params": self.model.get_shared_params(),
                "lr": self.lr,
                "name": "shared_encoder",
            },
            {
                "params": self.model.get_active_head_params(),
                "lr": self.lr * self.head_lr_mul,
                "name": "task_heads",
            },
        ]
        optimizer = AdamW(param_groups, weight_decay=0.01)

        warmup_steps = int(self.warmup_proportion * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        return optimizer, scheduler

    # ── Single Training Epoch ───────────────────────────────────────────

    def train_one_epoch(
        self,
        optimizer: AdamW,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        epoch: int,
        phase: str,
    ) -> float:
        """
        Run one full training epoch.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"[{phase}] Epoch {epoch + 1}",
            leave=False,
        )

        for inputs, labels in pbar:
            # ── Move to device ──────────────────────────────────────────
            input_ids = inputs["input_ids"].to(self.device)
            attn_mask = inputs["attention_mask"].to(self.device)
            labels_bias = labels["labels_bias"].to(self.device)
            labels_emotion = labels["labels_emotion"].to(self.device)

            # Social group labels only when not ablated
            labels_social = (
                labels["labels_social"].to(self.device)
                if not self.model.is_ablated
                else None
            )

            # ── Forward ─────────────────────────────────────────────────
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                labels_bias=labels_bias,
                labels_emotion=labels_emotion,
                labels_social=labels_social,
            )

            loss = outputs["loss"]

            # ── Backward ────────────────────────────────────────────────
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # ── Logging ─────────────────────────────────────────────────
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            self.writer.add_scalar("train/loss", loss.item(), self.global_step)
            if "loss_bias" in outputs:
                self.writer.add_scalar(
                    "train/loss_bias", outputs["loss_bias"].item(), self.global_step
                )
            if "loss_emotion" in outputs:
                self.writer.add_scalar(
                    "train/loss_emotion", outputs["loss_emotion"].item(), self.global_step
                )
            if "loss_social" in outputs:
                self.writer.add_scalar(
                    "train/loss_social", outputs["loss_social"].item(), self.global_step
                )

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    # ── Evaluation ──────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        tag: str = "val",
        epoch: int = 0,
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataloader, computing:
          - loss (combined)
          - pearson_r (bias regression quality)
          - emotion_jaccard (multi-label emotion quality)
          - social_accuracy (group classification quality)

        Returns:
            Dict of metric_name → value.
        """
        self.model.eval()

        all_loss = []
        all_bias_true, all_bias_pred = [], []
        all_emo_true, all_emo_pred = [], []
        all_soc_true, all_soc_pred = [], []

        for inputs, labels in tqdm(loader, desc=f"  [{tag}]", leave=False):
            input_ids = inputs["input_ids"].to(self.device)
            attn_mask = inputs["attention_mask"].to(self.device)
            labels_bias = labels["labels_bias"].to(self.device)
            labels_emotion = labels["labels_emotion"].to(self.device)
            labels_social = (
                labels["labels_social"].to(self.device)
                if not self.model.is_ablated
                else None
            )

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                labels_bias=labels_bias,
                labels_emotion=labels_emotion,
                labels_social=labels_social,
            )

            all_loss.append(outputs["loss"].item())
            all_bias_true.append(labels_bias)
            all_bias_pred.append(outputs["bias_score"].squeeze(-1))
            all_emo_true.append(labels_emotion)
            all_emo_pred.append(outputs["emotions"])

            if labels_social is not None:
                all_soc_true.append(labels_social)
                all_soc_pred.append(outputs["social_group"])

        # ── Aggregate ───────────────────────────────────────────────────
        metrics: Dict[str, float] = {}
        metrics["loss"] = float(np.mean(all_loss))
        metrics["pearson_r"] = pearson_correlation(
            torch.cat(all_bias_true), torch.cat(all_bias_pred)
        )
        metrics["emotion_jaccard"] = emotion_jaccard(
            torch.cat(all_emo_true), torch.cat(all_emo_pred)
        )
        if all_soc_true:
            metrics["social_accuracy"] = social_accuracy(
                torch.cat(all_soc_true), torch.cat(all_soc_pred)
            )

        # ── TensorBoard ────────────────────────────────────────────────
        for key, val in metrics.items():
            self.writer.add_scalar(f"{tag}/{key}", val, epoch)

        return metrics

    # ── Checkpoint ──────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, phase: str) -> None:
        path = self.output_dir / f"best_model_{phase}.pt"
        torch.save(
            {
                "epoch": epoch,
                "phase": phase,
                "model_state_dict": self.model.state_dict(),
                "best_val_loss": self.best_val_loss,
            },
            path,
        )
        log.info(f"  ✓ Checkpoint saved → {path}")

    def _save_predictions(self, loader: DataLoader, filename: str) -> None:
        """Run inference and save predictions to CSV."""
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in loader:
                input_ids = inputs["input_ids"].to(self.device)
                attn_mask = inputs["attention_mask"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attn_mask)
                all_preds.append(outputs["bias_score"].squeeze(-1).cpu())
                all_labels.append(labels["labels_bias"])

        preds = torch.cat(all_preds).numpy()
        labs = torch.cat(all_labels).numpy()

        path = self.output_dir / filename
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["prediction", "label"])
            writer.writerows(zip(preds.flatten(), labs.flatten()))
        log.info(f"  ✓ Predictions saved → {path}")

    # ── Main Training Loop ──────────────────────────────────────────────

    def run(self) -> None:
        """
        Execute the full two-phase training pipeline:

        Phase 1: Full MTL (all heads active)
            → Trains bias + emotion + social group heads concurrently.
            → Saves best checkpoint based on val loss.

        Phase 2: Ablation (social group head removed)
            → Loads best Phase 1 checkpoint.
            → Freezes social group head, zeros its output.
            → Fine-tunes bias + emotion heads only.
            → Saves best Phase 2 checkpoint.

        After both phases: runs test evaluation and saves predictions.
        """
        total_time_start = time.time()

        # ================================================================
        # PHASE 1: Full Multi-Task Learning
        # ================================================================
        log.info("=" * 70)
        log.info("PHASE 1: Full Multi-Task Learning (all 3 heads active)")
        log.info("=" * 70)

        total_steps_p1 = len(self.train_loader) * self.max_epochs_p1
        optimizer, scheduler = self._build_optimizer_and_scheduler(total_steps_p1)
        patience_counter = 0

        for epoch in range(self.max_epochs_p1):
            train_loss = self.train_one_epoch(
                optimizer, scheduler, epoch, "Phase1"
            )
            val_metrics = self.evaluate(self.val_loader, "val", epoch)

            log.info(
                f"  Phase1 Epoch {epoch + 1}/{self.max_epochs_p1} — "
                f"train_loss: {train_loss:.4f} | "
                f"val_loss: {val_metrics['loss']:.4f} | "
                f"pearson: {val_metrics['pearson_r']:.4f} | "
                f"emo_jaccard: {val_metrics['emotion_jaccard']:.4f} | "
                f"soc_acc: {val_metrics.get('social_accuracy', 0):.4f}"
            )

            # ── Early stopping & checkpointing ─────────────────────────
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self._save_checkpoint(epoch, "phase1")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    log.info(f"  ⚠ Early stopping at epoch {epoch + 1}")
                    break

        # ================================================================
        # PHASE 2: Ablation Fine-Tuning
        # ================================================================
        log.info("")
        log.info("=" * 70)
        log.info("PHASE 2: Ablation (social group head DISABLED)")
        log.info("=" * 70)

        # Load best Phase 1 checkpoint
        ckpt_path = self.output_dir / "best_model_phase1.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(ckpt["model_state_dict"])
            log.info(f"  ✓ Loaded Phase 1 best checkpoint (epoch {ckpt['epoch'] + 1})")

        # Activate ablation
        self.model.set_ablation_mode(ablate=True)

        # Rebuild optimizer (now excludes frozen social params)
        total_steps_p2 = len(self.train_loader) * self.max_epochs_p2
        optimizer, scheduler = self._build_optimizer_and_scheduler(total_steps_p2)
        self.best_val_loss = float("inf")  # reset for phase 2
        patience_counter = 0

        for epoch in range(self.max_epochs_p2):
            train_loss = self.train_one_epoch(
                optimizer, scheduler, epoch, "Phase2"
            )
            val_metrics = self.evaluate(self.val_loader, "val_ablated", epoch)

            log.info(
                f"  Phase2 Epoch {epoch + 1}/{self.max_epochs_p2} — "
                f"train_loss: {train_loss:.4f} | "
                f"val_loss: {val_metrics['loss']:.4f} | "
                f"pearson: {val_metrics['pearson_r']:.4f} | "
                f"emo_jaccard: {val_metrics['emotion_jaccard']:.4f}"
            )

            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self._save_checkpoint(epoch, "phase2")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    log.info(f"  ⚠ Early stopping at epoch {epoch + 1}")
                    break

        # ================================================================
        # TEST EVALUATION
        # ================================================================
        log.info("")
        log.info("=" * 70)
        log.info("FINAL TEST EVALUATION")
        log.info("=" * 70)

        # Load best Phase 2 checkpoint for final test
        ckpt_path = self.output_dir / "best_model_phase2.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(ckpt["model_state_dict"])
            log.info(f"  ✓ Loaded Phase 2 best checkpoint")

        test_metrics = self.evaluate(self.test_loader, "test", 0)

        log.info(
            f"  TEST RESULTS — "
            f"loss: {test_metrics['loss']:.4f} | "
            f"pearson: {test_metrics['pearson_r']:.4f} | "
            f"emo_jaccard: {test_metrics['emotion_jaccard']:.4f}"
        )

        self._save_predictions(self.test_loader, "test_predictions.csv")

        total_time = time.time() - total_time_start
        log.info(f"\n  Total training time: {total_time / 60:.1f} minutes")
        self.writer.close()
