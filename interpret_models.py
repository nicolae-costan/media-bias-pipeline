"""
interpret_models.py
-------------------
Performs token-level ablation (masking) analysis to find which words 
are most important for model predictions.

Supports:
1. Emotion Model (13 emotions)
2. Bias Model (Us vs Them regression)

Usage:
    python interpret_models.py --model_type emotion --checkpoint path/to/emotion.ckpt
    python interpret_models.py --model_type bias --checkpoint path/to/bias.ckpt
"""

import argparse
import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer
from collections import Counter
import re

# Ensure we can import from the subdirectories
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "EmotionModels"))
sys.path.append(os.path.join(project_root, "SentimentClassification"))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=["emotion", "bias"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument("--top_n_words", type=int, default=50, help="Number of frequent words to analyze")
    parser.add_argument("--num_samples", type=int, default=200, help="Number of sentences to test per word")
    parser.add_argument("--output_dir", type=str, default="interpret_results")
    return parser.parse_args()

def load_emotion_model(checkpoint_path):
    from EmotionModels.model import EmotionModel
    model = EmotionModel.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model.hparams.encoder_model)
    return model, tokenizer, "Emotion"

def load_bias_model(checkpoint_path):
    from SentimentClassification.BertRegression import BertRegressor
    # BertRegressor wraps the RedditTransformer
    model = BertRegressor.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model.hparams.model_name)
    return model, tokenizer, "Bias"

def get_frequent_words(df, text_col, top_n=50):
    """Find most frequent non-stop words."""
    stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "is", "are", "was", "were", "of"}
    all_text = " ".join(df[text_col].astype(str).tolist()).lower()
    words = re.findall(r'\b\w+\b', all_text)
    meaningful_words = [w for w in words if w not in stop_words and len(w) > 2]
    return [w for w, count in Counter(meaningful_words).most_common(top_n)]

def run_ablation(model, tokenizer, model_type, sentences, target_word):
    """
    Measures the average change in model output when target_word is masked.
    Positive value = the word was INCREASING the prediction score.
    """
    deltas = []
    
    for text in sentences:
        # Original score
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            if model_type == "emotion":
                out = torch.sigmoid(model(inputs['input_ids'], inputs['attention_mask']))
            else:
                out, _, _ = model.model(inputs)
                out = torch.sigmoid(out)
        
        orig_score = out.numpy().flatten()
        
        # Masked score
        masked_text = re.sub(rf'\b{target_word}\b', tokenizer.mask_token if tokenizer.mask_token else "[MASK]", text, flags=re.IGNORECASE)
        
        inputs_m = tokenizer(masked_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            if model_type == "emotion":
                out_m = torch.sigmoid(model(inputs_m['input_ids'], inputs_m['attention_mask']))
            else:
                out_m, _, _ = model.model(inputs_m)
                out_m = torch.sigmoid(out_m)
        
        masked_score = out_m.numpy().flatten()
        
        # Change = Original - Masked
        # If result is +0.3, it means the word "adds" 0.3 to the score.
        deltas.append(orig_score - masked_score)
        
    return np.mean(deltas, axis=0)

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Model
    if args.model_type == "emotion":
        model, tokenizer, name = load_emotion_model(args.checkpoint)
        dataset_path = "data/final_labels_MBIC.csv"
        text_col = "text"
        label_names = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Gratitude', 'Guilt', 'Happiness', 'Hope', 'Pride', 'Relief', 'Sadness', 'Sympathy', 'Neutral']
    else:
        model, tokenizer, name = load_bias_model(args.checkpoint)
        dataset_path = "data/UsVsThem_train_public.csv"
        text_col = "text"
        label_names = ["UsVsThem_Bias"]

    # 2. Load Data
    print(f"--- Analyzing {name} Model ---")
    df = pd.read_csv(dataset_path, sep=";" if "MBIC" in dataset_path else ",")
    frequent_words = get_frequent_words(df, text_col, args.top_n_words)
    print(f"Found {len(frequent_words)} words to test.")

    # 3. Analyze each word
    results = {}
    for word in tqdm(frequent_words, desc="Ablation progress"):
        samples = df[df[text_col].str.contains(rf'\b{word}\b', case=False, na=False, regex=True)][text_col].head(args.num_samples).tolist()
        if len(samples) < 5: continue
        
        contribution = run_ablation(model, tokenizer, args.model_type, samples, word)
        results[word] = contribution

    # 4. Save Data & Plot
    # Create a DataFrame for all results
    res_df = pd.DataFrame.from_dict(results, orient='index', columns=label_names)
    res_df.to_csv(os.path.join(args.output_dir, f"{args.model_type}_word_contributions.csv"))
    
    # Determine 'Overall Importance' for plotting
    # For Bias: it's the single score. For Emotion: it's the max impact across any emotion.
    if args.model_type == "bias":
        res_df['plot_val'] = res_df[label_names[0]]
    else:
        # For emotions, find the emotion each word impacts MOST
        res_df['plot_val'] = res_df.abs().max(axis=1)
        # Add a helper column showing which emotion that was
        res_df['top_emotion'] = res_df[label_names].abs().idxmax(axis=1)

    sorted_df = res_df.sort_values(by='plot_val', ascending=False).head(20)
    
    plt.figure(figsize=(12, 10))
    colors = ['salmon' if x > 0 else 'skyblue' for x in sorted_df['plot_val']]
    plt.barh(sorted_df.index[::-1], sorted_df['plot_val'][::-1], color=colors[::-1])
    
    title_suffix = ""
    if args.model_type == "emotion":
        title_suffix = "\n(Shows max impact across all 13 emotions)"
        # Annotate bars with the emotion name
        for i, (word, row) in enumerate(sorted_df.iloc[::-1].iterrows()):
            plt.text(row['plot_val'], i, f"  ({row['top_emotion']})", va='center', fontsize=9)

    plt.xlabel("Average Contribution to Prediction Score (Orig - Masked)")
    plt.title(f"Top 20 Most Influential Words for {name} Model{title_suffix}")
    plt.axvline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    
    plot_path = os.path.join(args.output_dir, f"{args.model_type}_importance.png")
    plt.savefig(plot_path)
    print(f"\n--- Analysis Complete ---")
    print(f"Plot saved to: {plot_path}")
    print(f"Detailed CSV saved to: {os.path.join(args.output_dir, f'{args.model_type}_word_contributions.csv')}")


if __name__ == "__main__":
    main()
