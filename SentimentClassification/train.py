"""
Runs a model on a single node across N-gpus using TensorBoard for local logging.
"""
import os
from pathlib import Path

# Ensure this matches your filename: BertRegression.py
from BertRegression import BERTRegressor as BERTClassifier
from pytorch_lightning import Trainer, seed_everything # Use built-in seed
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from test_tube import HyperOptArgumentParser

def main(hparams) -> None:
    """
    Main training routine.
    """
    # FIX: Replaced torchnlp set_seed with Lightning seed_everything
    seed_everything(hparams.seed)

    # ------------------------
    # 1. INIT LIGHTNING MODEL
    # ------------------------
    model = BERTClassifier(hparams)

    # ------------------------
    # 2. INIT LOGGER (Local TensorBoard)
    # ------------------------
    tb_logger = TensorBoardLogger(
        save_dir="tb_logs",
        name=f"task_{hparams.aux_task}"
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
    # Note: versioning is handled automatically by tb_logger
    ckpt_path = os.path.join(tb_logger.log_dir, "checkpoints")
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,  # FIX: Changed 'filepath' to 'dirpath'
        filename='{epoch}-{val_loss:.2f}',  # Optional: adds info to the filename
        save_top_k=hparams.save_top_k,
        verbose=True,
        monitor=hparams.monitor,
        # period=1,               # Note: 'period' is now 'every_n_epochs' in newer versions
        mode=hparams.metric_mode,
    )

    model.hparams.checkpoint_path = tb_logger.log_dir

    # ------------------------
    # 4. INIT TRAINER
    # ------------------------
    # ------------------------
    # 4. INIT TRAINER
    # ------------------------
    trainer = Trainer(
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],

        # THE FIX: 'gpus' is now split into accelerator and devices
        accelerator="auto",  # Tells Lightning to use the GPU
        devices=hparams.gpus,  # Tells it how many to use (e.g., 1)

        # THE FIX: 'dp' (DataParallel) is mostly deprecated;
        # 'ddp' (DistributedDataParallel) is the modern standard.
        strategy="ddp" if hparams.gpus > 1 else "auto",

        max_epochs=hparams.max_epochs,
        min_epochs=hparams.min_epochs,
        accumulate_grad_batches=hparams.accumulate_grad_batches,

        # THE FIX: 'val_percent_check' was renamed to 'limit_val_batches'
        limit_val_batches=hparams.val_percent_check,
    )
    # ------------------------
    # 5. START TRAINING
    # ------------------------
    trainer.fit(model)
    trainer.test()

if __name__ == "__main__":
    parser = HyperOptArgumentParser(
        strategy="random_search",
        description="BERT Multi-task Regressor",
        add_help=True,
    )

    # General Setup
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--save_top_k", default=1, type=int)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--min_epochs", default=1, type=int)
    parser.add_argument("--max_epochs", default=10, type=int)

    # Monitoring Settings
    parser.add_argument("--monitor", default="val_loss", type=str)
    parser.add_argument("--metric_mode", default="min", type=str)

    # Hardware & Batching
    parser.add_argument("--batch_size", default=6, type=int)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--accumulate_grad_batches", default=2, type=int)
    parser.add_argument("--val_percent_check", default=1.0, type=float)

    # Search mode
    parser.add_argument("--search_mode", default=False, type=bool)

    # Add BERT-specific arguments from the model class
    parser = BERTClassifier.add_model_specific_args(parser)
    hparams = parser.parse_args()

    if not hparams.search_mode:
        main(hparams)
    else:
        for hparam_trial in hparams.trials(8):
            main(hparam_trial)