"""
Quick training test using existing code
"""
import sys
sys.path.insert(0, '/home/nicu/facultate/big_data/media-bias-pipeline/SentimentClassification')

import os
from pathlib import Path

# Set paths
os.chdir('/home/nicu/facultate/big_data/media-bias-pipeline/SentimentClassification')

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from test_tube import HyperOptArgumentParser

# Import after path setup
from BertRegression import BERTRegressor as BERTClassifier

# Create hparams manually for quick test
class SimpleNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

hparams = SimpleNamespace(
    seed=42,
    encoder_model="bert-base-uncased",  # Using bert-base for faster test
    max_epochs=1,
    min_epochs=1,
    batch_size=4,
    gpus=0,  # CPU only for test
    accumulate_grad_batches=1,
    encoder_learning_rate=1e-5,
    aux_task="emotions",  # Test with emotions auxiliary task
    gradnorm=False,
    loss_aux_dropout=0.25,
    extra_dropout=0.1,
    nr_frozen_epochs=0,
    max_length=128,  # Shorter for quick test
    warmup_proportion=0.1,
    patience=3,
    save_top_k=1,
    monitor="val_loss",
    metric_mode="min",
    val_percent_check=0.5,  # Only use half of validation
    loader_workers=0,  # No multiprocessing for test
    # Data paths - updated to absolute paths
    train_csv="/home/nicu/facultate/big_data/Reddit UsVsThem Data/UsVsThem_train_public.csv",
    dev_csv="/home/nicu/facultate/big_data/Reddit UsVsThem Data/UsVsThem_valid_public.csv",
    test_csv="/home/nicu/facultate/big_data/Reddit UsVsThem Data/UsVsThem_test_public.csv",
)

print("=" * 60)
print("QUICK TRAINING TEST")
print("=" * 60)
print(f"Model: {hparams.encoder_model}")
print(f"Auxiliary task: {hparams.aux_task}")
print(f"Epochs: {hparams.max_epochs}")
print(f"Batch size: {hparams.batch_size}")
print("=" * 60)

# Initialize
print("\n[1/4] Initializing model...")
seed_everything(hparams.seed)
model = BERTClassifier(hparams)
print("✓ Model initialized")

# Logger
print("\n[2/4] Setting up logger...")
tb_logger = TensorBoardLogger(
    save_dir="/home/nicu/facultate/big_data/media-bias-pipeline/tb_logs",
    name=f"test_{hparams.aux_task}"
)
hparams.checkpoint_path = tb_logger.log_dir
print(f"✓ Logger ready at: {tb_logger.log_dir}")

# Callbacks
print("\n[3/4] Setting up callbacks...")
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=hparams.patience,
    verbose=True,
    mode="min",
)
checkpoint = ModelCheckpoint(
    dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
    filename='{epoch}-{val_loss:.2f}',
    save_top_k=1,
    verbose=True,
    monitor="val_loss",
    mode="min",
)
print("✓ Callbacks ready")

# Trainer
print("\n[4/4] Initializing trainer...")
trainer = Trainer(
    logger=tb_logger,
    callbacks=[checkpoint, early_stop],
    accelerator="cpu",
    devices=1,
    max_epochs=hparams.max_epochs,
    min_epochs=hparams.min_epochs,
    accumulate_grad_batches=hparams.accumulate_grad_batches,
    limit_val_batches=hparams.val_percent_check,
    enable_progress_bar=True,
)
print("✓ Trainer ready")

# Run training
print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)
trainer.fit(model)

print("\n" + "=" * 60)
print("TRAINING COMPLETE - Running test...")
print("=" * 60)
trainer.test()

print("\n" + "=" * 60)
print("TEST RUN SUCCESSFUL!")
print("=" * 60)
