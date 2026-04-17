"""
Standalone Testing Script for RoBERTa version_3 Checkpoint
Usage: python test_roberta_v3.py
"""
import os
import sys
import torch
import yaml
from pathlib import Path
from argparse import Namespace
from pytorch_lightning import Trainer, seed_everything
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------
# 1. SETUP & FIXES
# ---------------------------------------------------------

# Add the SentimentClassification directory to sys.path
BASE_DIR = Path("/media-bias-pipeline")
sys.path.append(str(BASE_DIR / "SentimentClassification"))

from RoBERTaRegression import RoBERTaRegressor

# Modern PyTorch (2.6+) FIX: Allow LabelEncoder to be unpickled from hparams
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([LabelEncoder])

def main():
    # ---------------------------------------------------------
    # 2. DEFINE PATHS
    # ---------------------------------------------------------
    # Change working directory to BASE_DIR so relative paths in hparams work
    os.chdir(str(BASE_DIR))
    
    ckpt_path = "/media-bias-pipeline/tb_logs/task_None_roberta/version_3/checkpoints/epoch=5-val_loss=0.12.ckpt"
    hparams_path = "/media-bias-pipeline/tb_logs/task_None_roberta/version_3/hparams.yaml"
    
    print(f"--- Loading RoBERTa Checkpoint: {ckpt_path} ---")
    
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        return

    # ---------------------------------------------------------
    # 3. LOAD HPARAMS & MODEL
    # ---------------------------------------------------------
    with open(hparams_path, 'r') as f:
        hparams_data = yaml.safe_load(f)
    
    # Ensure hparams has all necessary fields
    hparams = Namespace(**hparams_data)
    
    # Fix paths in hparams if they are relative to the old execution root
    # (assuming they were relative to media-bias-pipeline/)
    # They should already be 'Resources/...' so they work after os.chdir(BASE_DIR)
    
    # Set seed for reproducibility
    seed_everything(hparams.seed)

    try:
        # Load the model with weights_only=False to bypass security restrictions on trusted local file
        model = RoBERTaRegressor.load_from_checkpoint(
            ckpt_path, 
            hparams=hparams, 
            weights_only=False
        )
        print("--- Model Loaded Successfully ---")
    except Exception as e:
        print(f"CRITICAL ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Ensure the model knows where to save outputs (predictions.csv)
    model.hparams.checkpoint_path = str(Path(ckpt_path).parent.parent)
    print(f"--- Predictions will be saved to: {model.hparams.checkpoint_path}/predictions.csv ---")

    # ---------------------------------------------------------
    # 4. INIT TRAINER & RUN TEST
    # ---------------------------------------------------------
    trainer = Trainer(
        logger=False,           # No need for logs during pure testing
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
    )

    print("--- Starting Test Loop ---")
    trainer.test(model)
    print("--- Testing Complete ---")

if __name__ == "__main__":
    main()
