import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns

# Add project root to path so we can import backend
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import get_settings
from backend.model_service import ModelService

def main():
    print("Loading CSV...")
    csv_path = "babe_word_saliency_statistics.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    df_filtered = df[df["occurrences"] >= 3].copy()
    
    categories = [
        "avg_bias_drop", "avg_anger_drop", "avg_disgust_drop", 
        "avg_fear_drop", "avg_joy_drop", "avg_optimism_drop", 
        "avg_sadness_drop", "avg_neutral_drop"
    ]
    
    # Collect top 15 words for each category by absolute drop
    target_words = set()
    word_to_category = {}
    
    for cat in categories:
        df_filtered['abs_' + cat] = df_filtered[cat].abs()
        top_words = df_filtered.sort_values(by='abs_' + cat, ascending=False).head(15)['word'].tolist()
        
        display_cat = cat.replace('avg_', '').replace('_drop', '').title()
        for w in top_words:
            target_words.add(w)
            if w not in word_to_category:
                word_to_category[w] = display_cat
                
    target_words = list(target_words)
    print(f"Collected {len(target_words)} unique highly influential words to map.")
    
    print("Loading ModelService...")
    # NOTE TO USER: 
    # Make sure you have an .env file in the root directory (c:\Users\rares\Desktop\BigData\media-bias-pipeline)
    # with BIAS_CHECKPOINT_PATH set to your actual checkpoint file.
    settings = get_settings()
    model_service = ModelService(settings)
    model_service.load()
    
    if model_service.bias_model is None:
        print("ERROR: Bias model failed to load. Please check your .env file and ensure BIAS_CHECKPOINT_PATH is correct.")
        return

    bias_model = model_service.bias_model
    tokenizer = model_service.bias_tokenizer
    device = model_service.device
    
    # Extract embeddings
    print("Extracting embeddings...")
    embeddings = []
    valid_words = []
    valid_categories = []
    
    try:
        # Get the actual huggingface model from the PyTorch Lightning wrapper
        hf_model = bias_model.model
        word_embeddings_layer = hf_model.get_input_embeddings()
    except Exception as e:
        print("Could not get input embeddings layer directly:", e)
        return

    with torch.no_grad():
        for word in target_words:
            # Tokenize word without special tokens
            tokens = tokenizer(word, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
            
            if tokens.shape[1] == 0:
                continue
                
            # Get embeddings
            word_emb = word_embeddings_layer(tokens)
            
            # If word is split into multiple subwords, take the mean
            mean_emb = word_emb.mean(dim=1).squeeze().cpu().numpy()
            
            embeddings.append(mean_emb)
            valid_words.append(word)
            valid_categories.append(word_to_category[word])
            
    embeddings = np.array(embeddings)
    
    print("Reducing dimensionality with PCA + t-SNE...")
    n_samples = len(embeddings)
    
    if n_samples > 50:
        pca = PCA(n_components=50)
        reduced = pca.fit_transform(embeddings)
    else:
        reduced = embeddings
        
    perplexity = min(30, n_samples - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
    coords_2d = tsne.fit_transform(reduced)
    
    print("Plotting...")
    plt.figure(figsize=(16, 12))
    
    sns.scatterplot(
        x=coords_2d[:, 0], 
        y=coords_2d[:, 1],
        hue=valid_categories,
        palette="tab10",
        s=150,
        alpha=0.8,
        edgecolor="w"
    )
    
    # Add labels
    for i, word in enumerate(valid_words):
        plt.annotate(
            word,
            (coords_2d[i, 0], coords_2d[i, 1]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=10,
            alpha=0.8
        )
        
    plt.title("2D Map of Bias-Influential Words\n(Extracted from Fine-Tuned Bias Model Embeddings)", fontsize=16)
    plt.legend(title="Dominant Category", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    out_path = "bias_word_map.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Successfully saved plot to {out_path}")

if __name__ == "__main__":
    main()
