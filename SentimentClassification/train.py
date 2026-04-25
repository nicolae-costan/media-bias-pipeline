"""
Runs a model on a single node across N-gpus using TensorBoard for local logging.
"""
import torch
import os
from pathlib import Path

# Ensure this matches your filename: BertRegression.py
from BertRegression import BERTRegressor as BERTClassifier
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
    # EarlyStopping: halt when the emotion Jaccard score stops improving.
    # Using val_acc_aux (epoch-level macro Jaccard over all 13 emotions) in
    # 'max' mode means we keep training as long as the model keeps getting
    # better at predicting emotions, and stop when it plateaus.
    early_stop_callback = EarlyStopping(
        monitor=hparams.monitor,
        patience=hparams.patience,
        verbose=True,
        mode=hparams.metric_mode,
    )

    # Checkpoint: Save the epoch with the HIGHEST emotion Jaccard score.
    # This ensures .test() loads the model that was best at emotion prediction,
    # not just the one that happened to have the lowest MSE.
    ckpt_path = os.path.join(tb_logger.log_dir, "checkpoints")
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        filename='epoch={epoch}-val_acc_aux={val_acc_aux:.4f}',
        save_top_k=hparams.save_top_k,
        verbose=True,
        monitor=hparams.monitor,
        mode=hparams.metric_mode,
        auto_insert_metric_name=False,  # filename template above already includes it
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
    # val_acc_aux = epoch-level macro Jaccard over all 13 emotions (maximize).
    # Switch to 'val_pearson' / 'max' if the primary goal shifts to the
    # main regression task, or 'val_loss' / 'min' for generic MSE monitoring.
    parser.add_argument("--monitor", default="val_acc_aux", type=str)
    parser.add_argument("--metric_mode", default="max", type=str)

    # Model Selection
    parser.opt_list(
        "--model_type",
        default="bert",
        tunable=False,
        type=str,
        options=["bert"],
        help="The architecture to use for training (currently only BERT is supported)",
    )

    # Hardware & Batching
    parser.add_argument("--batch_size", default=6, type=int)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--accumulate_grad_batches", default=2, type=int)
    parser.add_argument("--val_percent_check", default=1.0, type=float)

    # Search mode
    parser.add_argument("--search_mode", default=False, type=bool)

    # Add model-specific arguments
    parser = BERTClassifier.add_model_specific_args(parser)
    hparams = parser.parse_args()

    # Set default encoder model based on type (currently always bert-base-uncased)
    if hparams.encoder_model == "roberta-base":
        print("WARNING: RoBERTa backbone requested but RoBERTa classes were removed. Falling back to bert-base-uncased.")
        hparams.encoder_model = "bert-base-uncased"

    if not hparams.search_mode:
        main(hparams)
    else:
        for hparam_trial in hparams.trials(8):
            main(hparam_trial)