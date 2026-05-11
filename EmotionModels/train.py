"""
Runs a model on a single node across N-gpus using TensorBoard for local logging.
"""
import os
from pathlib import Path

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from test_tube import HyperOptArgumentParser

# --- THE FIX: Import your new 13-Emotion model ---
from model import EmotionModel


def main(hparams) -> None:
    """
    Main training routine.
    """
    seed_everything(hparams.seed)

    # ------------------------
    # 1. INIT LIGHTNING MODEL
    # ------------------------
    # --- THE FIX: Instantiate the new model ---
    model = EmotionModel(hparams)

    # ------------------------
    # 2. INIT LOGGER (Local TensorBoard)
    # ------------------------
    # --- THE FIX: Removed aux_task since we are only predicting emotions now ---
    tb_logger = TensorBoardLogger(
        save_dir="tb_logs",
        name="emotion_classification"
    )

    # ------------------------
    # 3. INIT CALLBACKS
    # ------------------------
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=hparams.patience,
        verbose=True,
        mode="min",
    )

    # Checkpoint: Save best model version
    ckpt_path = os.path.join(tb_logger.log_dir, "checkpoints")
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        filename='{epoch}-{val_loss:.4f}',
        save_top_k=hparams.save_top_k,
        verbose=True,
        monitor=hparams.monitor,
        mode=hparams.metric_mode,
    )

    # Lightning handles paths better now, but we'll leave this for compatibility
    if not hasattr(model.hparams, 'checkpoint_path'):
        model.hparams.checkpoint_path = tb_logger.log_dir

    # ------------------------
    # 4. INIT TRAINER
    # ------------------------
    trainer = Trainer(
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="auto",
        devices=hparams.gpus,
        strategy="ddp" if hparams.gpus > 1 else "auto",
        max_epochs=hparams.max_epochs,
        min_epochs=hparams.min_epochs,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        gradient_clip_val=hparams.grad_clip,  # Threshold for clipping
        gradient_clip_algorithm="norm",
        limit_val_batches=hparams.val_percent_check,
    )

    # ------------------------
    # 5. START TRAINING
    # ------------------------
    trainer.fit(model)
    trainer.test(model)  # --- THE FIX: Explicitly pass the model to test() ---


if __name__ == "__main__":
    parser = HyperOptArgumentParser(
        strategy="random_search",
        description="RoBERTa Emotion Classifier",
        add_help=True,
    )

    # General Setup
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--save_top_k", default=1, type=int)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--min_epochs", default=1, type=int)
    parser.add_argument("--max_epochs", default=10, type=int)

    # Monitoring Settings (You can also change this to "val_jaccard" and mode to "max" if you want)
    parser.add_argument("--monitor", default="val_loss", type=str)
    parser.add_argument("--metric_mode", default="min", type=str)

    # Hardware & Batching
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--accumulate_grad_batches", default=1, type=int)
    parser.add_argument("--val_percent_check", default=1.0, type=float)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    # Search mode
    parser.add_argument("--search_mode", default=False, type=bool)

    # --- THE FIX: Load args from the EmotionRegressor instead of BERTClassifier ---
    parser = EmotionModel.add_model_specific_args(parser)

    hparams = parser.parse_args()

    if not hparams.search_mode:
        main(hparams)
    else:
        for hparam_trial in hparams.trials(8):
            main(hparam_trial)