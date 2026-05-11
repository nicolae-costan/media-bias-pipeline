import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import jaccard_score
from torch.utils.data import DataLoader
from model import EmotionModel
from dataloader import sentiment_analysis_dataset, MyCollator

def optimize_thresholds(checkpoint_path, dev_csv, output_json="EmotionModels/thresholds.json"):
    print(f"--- Loading model from: {checkpoint_path} ---")
    
    # 1. Load Model
    # We load the checkpoint manually first to get the saved hparams
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    hparams = checkpoint.get("hyper_parameters", {})
    
    # Instantiate model with the saved hparams
    model = EmotionModel.load_from_checkpoint(checkpoint_path, hparams=hparams, strict=False)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    # 2. Step A: Inference Cache (Collect probs and targets)
    print("--- Step A: Collecting probabilities and targets from validation set ---")
    
    # Setup DataLoader for Dev set
    model.hparams.dev_csv = dev_csv
    dataset = sentiment_analysis_dataset(model.hparams, train=False, val=True, test=False)
    dataloader = DataLoader(
        dataset, 
        batch_size=model.hparams.batch_size, 
        collate_fn=model.prepare_sample,
        num_workers=0 # Keep it simple for caching
    )

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs, targets = batch
            input_ids, attention_mask = model._safe_squeeze(inputs)
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()

            logits_28 = model.forward(input_ids, attention_mask)
            probs_28 = torch.sigmoid(logits_28)
            
            # Expand targets_13 to targets_28 using the model's mapping
            targets_13 = targets['labels_aux']
            if torch.cuda.is_available():
                targets_13 = targets_13.cuda()
            
            targets_28 = targets_13[:, model.mapping]

            all_probs.append(probs_28.cpu().numpy())
            all_targets.append(targets_28.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # 3. Step B: Data Splitting (50/50 split)
    print("--- Step B: Splitting data into Calibration and Internal Test ---")
    n_samples = len(probs)
    indices = np.arange(n_samples)
    np.random.seed(42) # For reproducibility
    np.random.shuffle(indices)
    
    split_idx = n_samples // 2
    calib_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    probs_calib, targets_calib = probs[calib_indices], targets[calib_indices]
    probs_test, targets_test = probs[test_indices], targets[test_indices]

    # 4. Step C: Per-Class Grid Search
    print("--- Step C: Finding optimal thresholds per class ---")
    best_thresholds = []
    
    # There are 28 nodes
    for i in range(28):
        best_t = 0.5
        best_jaccard = -1.0
        
        y_true = targets_calib[:, i]
        y_prob = probs_calib[:, i]
        
        # Test 100 thresholds from 0.01 to 0.99
        for t in np.linspace(0.01, 0.99, 99):
            y_pred = (y_prob > t).astype(float)
            score = jaccard_score(y_true, y_pred, zero_division=0)
            
            if score > best_jaccard:
                best_jaccard = score
                best_t = t
        
        best_thresholds.append(float(best_t))
        print(f"Class {i:2}: Best T = {best_t:.2f} | Jaccard = {best_jaccard:.4f}")

    # 5. Step D: Final Evaluation
    print("--- Step D: Final Evaluation on Internal Test Set ---")
    
    # Baseline (0.5)
    baseline_preds = (probs_test > 0.5).astype(float)
    baseline_jaccard = jaccard_score(targets_test, baseline_preds, average="macro", zero_division=0)
    
    # Tuned
    tuned_preds = np.zeros_like(probs_test)
    for i in range(28):
        tuned_preds[:, i] = (probs_test[:, i] > best_thresholds[i]).astype(float)
    
    tuned_jaccard = jaccard_score(targets_test, tuned_preds, average="macro", zero_division=0)
    
    print("\n" + "="*30)
    print(f"Baseline Macro Jaccard (0.5): {baseline_jaccard:.4f}")
    print(f"Tuned Macro Jaccard:          {tuned_jaccard:.4f}")
    print(f"Improvement:                  {tuned_jaccard - baseline_jaccard:.4f}")
    print("="*30)

    # 6. Export
    with open(output_json, 'w') as f:
        json.dump(best_thresholds, f, indent=4)
    print(f"--- Exported thresholds to {output_json} ---")

if __name__ == "__main__":
    CKPT = "tb_logs/emotion_classification/version_4/checkpoints/epoch=3-val_loss=0.0858.ckpt"
    DEV_CSV = "EmotionModels/Resources/UsVsThem_valid_public.csv"
    
    # Check if we are running from project root or EmotionModels folder
    if not os.path.exists(CKPT) and os.path.exists("../" + CKPT):
        os.chdir("../") # Move to root
        
    optimize_thresholds(CKPT, DEV_CSV)
