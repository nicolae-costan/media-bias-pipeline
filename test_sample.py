"""
Quick training test on SAMPLE data only
"""
import sys
sys.path.insert(0, '/home/nicu/facultate/big_data/media-bias-pipeline/SentimentClassification')

# Fix PyTorch 2.6 weights_only issue
import torch
from sklearn.preprocessing import LabelEncoder
torch.serialization.add_safe_globals([LabelEncoder])

import os
os.chdir('/home/nicu/facultate/big_data/media-bias-pipeline/SentimentClassification')

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from test_tube import HyperOptArgumentParser

from BertRegression import BERTRegressor as BERTClassifier

class SimpleNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

hparams = SimpleNamespace(
    seed=42,
    encoder_model="bert-base-uncased",
    max_epochs=1,
    min_epochs=1,
    batch_size=4,  # Can use larger batch with GPU
    gpus=1,
    accumulate_grad_batches=1,
    encoder_learning_rate=1e-5,
    aux_task="emotions",
    gradnorm=False,
    loss_aux_dropout=0.25,
    extra_dropout=0.1,
    nr_frozen_epochs=0,
    max_length=64,  # Short for speed
    warmup_proportion=0.1,
    patience=2,
    save_top_k=1,
    monitor="val_loss",
    metric_mode="min",
    val_percent_check=1.0,
    loader_workers=0,
    # Sample data paths
    train_csv="/home/nicu/facultate/big_data/media-bias-pipeline/sample_train.csv",
    dev_csv="/home/nicu/facultate/big_data/media-bias-pipeline/sample_dev.csv",
    test_csv="/home/nicu/facultate/big_data/media-bias-pipeline/sample_test.csv",
)

print("=" * 60)
print("SAMPLE DATA TRAINING TEST (9 train, 5 dev, 5 test)")
print("=" * 60)
print(f"Model: {hparams.encoder_model}")
print(f"Aux task: {hparams.aux_task}")
print(f"Batch size: {hparams.batch_size}, Max length: {hparams.max_length}")
print("=" * 60)

print("\n[1/4] Setting up logger...")
tb_logger = TensorBoardLogger(
    save_dir="/home/nicu/facultate/big_data/media-bias-pipeline/tb_logs",
    name="sample_test"
)
print(f"✓ Logger at: {tb_logger.log_dir}")

print("\n[2/4] Initializing model...")
seed_everything(hparams.seed)
hparams.checkpoint_path = tb_logger.log_dir
model = BERTClassifier(hparams)
print("✓ Model initialized")

print("\n[3/4] Setting up callbacks...")
early_stop = EarlyStopping(monitor="val_loss", patience=hparams.patience, mode="min")
checkpoint = ModelCheckpoint(
    dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
    filename='{epoch}-{val_loss:.2f}',
    save_top_k=1,
    monitor="val_loss",
    mode="min",
)
print("✓ Callbacks ready")

print("\n[4/4] Initializing trainer...")
trainer = Trainer(
    logger=tb_logger,
    callbacks=[checkpoint, early_stop],
    accelerator="gpu",
    devices=1,
    max_epochs=hparams.max_epochs,
    min_epochs=1,
    enable_progress_bar=True,
)
print("✓ Trainer ready")

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)
trainer.fit(model)

print("\n" + "=" * 60)
print("TRAINING COMPLETE - Testing...")
print("=" * 60)
trainer.test()

print("\n" + "=" * 60)
print("SAMPLE TEST RUN SUCCESSFUL!")
print("=" * 60)
