"""
Runs a model on a single node across N-gpus using TensorBoard for local logging.
"""
import torch
import os
from pathlib import Path

# Ensure this matches your filename: BertRegression.py
from BertRegression import BERTRegressor as BERTClassifier
from RoBERTaRegression import RoBERTaRegressor as RoBERTaClassifier
from pytorch_lightning import Trainer, seed_everything # Use built-in seed
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from test_tube import HyperOptArgumentParser
from sklearn.preprocessing import LabelEncoder

# Modern PyTorch (2.6+) FIX: Allow LabelEncoder to be unpickled from hparams
# during checkpoint loading (otherwise .test() crashes with UnpicklingError)
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([LabelEncoder])

def main(hparams) -> None:
    """
    Main training routine.
    """
    # Fix for hardware precision/performance on Ampere+ GPUs
    torch.set_float32_matmul_precision('high')

    # FIX: Replaced torchnlp set_seed with Lightning seed_everything
    seed_everything(hparams.seed)

    # 1. INIT LIGHTNING MODEL
    # ------------------------
    if hparams.model_type == "roberta":
        print(f"--- Using RoBERTa Backbone ({hparams.encoder_model}) ---")
        model = RoBERTaClassifier(hparams)
    else:
        print(f"--- Using BERT Backbone ({hparams.encoder_model}) ---")
        model = BERTClassifier(hparams)

    # ------------------------
    # 2. INIT LOGGER (Local TensorBoard)
    # ------------------------
    tb_logger = TensorBoardLogger(
        save_dir="tb_logs",
        name=f"task_{hparams.aux_task}_{hparams.model_type}"
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
        accelerator="gpu" if hparams.gpus > 0 else "cpu",
        devices=hparams.gpus if hparams.gpus > 0 else 1,

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

    # Model Selection
    parser.opt_list(
        "--model_type",
        default="bert",
        tunable=False,
        type=str,
        options=["bert", "roberta"],
        help="The architecture to use for training",
    )

    # Hardware & Batching
    parser.add_argument("--batch_size", default=6, type=int)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--accumulate_grad_batches", default=2, type=int)
    parser.add_argument("--val_percent_check", default=1.0, type=float)

    # Search mode
    parser.add_argument("--search_mode", default=False, type=bool)

    # Add model-specific arguments
    # If using RoBERTa, it will still use similar arguments for now
    parser = BERTClassifier.add_model_specific_args(parser)
    hparams = parser.parse_args()

    # Set default encoder model based on type if not explicitly provided
    # (Note: HyperOptArgumentParser handle defaults, but we ensure consistency here)
    if hparams.model_type == "roberta" and hparams.encoder_model == "bert-base-uncased":
        hparams.encoder_model = "roberta-base"

    if not hparams.search_mode:
        main(hparams)
    else:
        for hparam_trial in hparams.trials(8):
            main(hparam_trial)