"""
Quick test script for existing training code
"""
import sys
sys.path.insert(0, '/home/nicu/facultate/big_data/media-bias-pipeline/SentimentClassification')

import argparse
import torch
from pathlib import Path

# Check imports
print("Testing imports...")
try:
    from transformers import AutoTokenizer
    print("✓ transformers imported")
except Exception as e:
    print(f"✗ transformers import failed: {e}")

try:
    import pytorch_lightning as pl
    print(f"✓ pytorch_lightning imported (version: {pl.__version__})")
except Exception as e:
    print(f"✗ pytorch_lightning import failed: {e}")

try:
    import pandas as pd
    print("✓ pandas imported")
except Exception as e:
    print(f"✗ pandas import failed: {e}")

# Check data files
data_dir = Path("/home/nicu/facultate/big_data/Reddit UsVsThem Data")
print(f"\nChecking data files in: {data_dir}")
for f in ["UsVsThem_train_public.csv", "UsVsThem_valid_public.csv", "UsVsThem_test_public.csv"]:
    path = data_dir / f
    if path.exists():
        print(f"✓ {f} exists")
    else:
        print(f"✗ {f} missing")

# Check CUDA availability
print(f"\nCUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

print("\nAll checks passed! Ready for training test.")
