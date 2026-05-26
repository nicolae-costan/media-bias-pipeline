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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    print("Loading CSV...")
    csv_path = os.path.join(script_dir, "babe_word_saliency_statistics.csv")
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
    
    print("Loading ModelService...")
    settings = get_settings()
    
    # Explicitly set correct paths for checkpoints to avoid issues
    settings.bias_checkpoint_path = os.path.join(project_root, "Checkpoints", "version_4", "epoch=1-val_f1_macro=0.8119.ckpt")
    settings.emotion_checkpoint_path = os.path.join(project_root, "Checkpoints", "version_2", "epoch=12-val_loss=0.1126.ckpt")
    settings.emotion_thresholds_path = os.path.join(project_root, "EmotionModels", "thresholds.json")
    
    model_service = ModelService(settings)
    model_service.load()
    
    if model_service.bias_model is None or model_service.emotion_model is None:
        print("ERROR: Models failed to load. Please check your checkpoints directory.")
        return

    device = model_service.device
    
    print("Extracting full model vocabulary embeddings...")
    try:
        bias_hf_model = model_service.bias_model.model
        bias_emb_layer = bias_hf_model.get_input_embeddings()
        bias_full_emb = bias_emb_layer.weight.detach()
        bias_full_norm = torch.nn.functional.normalize(bias_full_emb, p=2, dim=1)
        
        emo_hf_model = model_service.emotion_model.model
        emo_emb_layer = emo_hf_model.get_input_embeddings()
        emo_full_emb = emo_emb_layer.weight.detach()
        emo_full_norm = torch.nn.functional.normalize(emo_full_emb, p=2, dim=1)
    except Exception as e:
        print("Could not get input embeddings layer directly:", e)
        return

    for cat in categories:
        print(f"\nProcessing category: {cat}")
        df_filtered['abs_' + cat] = df_filtered[cat].abs()
        top_words = df_filtered.sort_values(by='abs_' + cat, ascending=False).head(10)['word'].tolist()
        
        display_cat = cat.replace('avg_', '').replace('_drop', '').title()
        
        if cat == "avg_bias_drop":
            model_name = "Bias"
            tokenizer = model_service.bias_tokenizer
            emb_layer = bias_emb_layer
            full_emb = bias_full_emb
            full_norm = bias_full_norm
        else:
            model_name = "Emotion"
            tokenizer = model_service.emotion_tokenizer
            emb_layer = emo_emb_layer
            full_emb = emo_full_emb
            full_norm = emo_full_norm
            
        category_embeddings = []
        category_words = []
        is_original = []
        
        with torch.no_grad():
            for word in top_words:
                # Tokenize word without special tokens
                tokens = tokenizer(word, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
                
                if tokens.shape[1] == 0:
                    continue
                    
                # Get embedding and mean-pool subwords
                word_emb = emb_layer(tokens).mean(dim=1).squeeze()
                
                category_embeddings.append(word_emb.cpu().numpy())
                category_words.append(word)
                is_original.append(True)
                
                # Find closest neighbors in model vocabulary
                word_emb_norm = torch.nn.functional.normalize(word_emb, p=2, dim=0)
                cos_sim = torch.matmul(full_norm, word_emb_norm)
                
                # Retrieve top 50 to filter out duplicates/special tokens
                top_k = torch.topk(cos_sim, k=50)
                
                neighbors_found = 0
                for idx in top_k.indices:
                    idx_val = idx.item()
                    if idx_val in tokenizer.all_special_ids:
                        continue
                        
                    token_str = tokenizer.convert_ids_to_tokens(idx_val)
                    # Clean RoBERTa subword spaces and padding
                    clean_token = token_str.replace('Ġ', '').replace(' ', '').lower()
                    
                    if len(clean_token) < 2:
                        continue
                    if clean_token == word.lower():
                        continue
                    if clean_token in category_words:
                        continue
                        
                    neighbors_found += 1
                    category_embeddings.append(full_emb[idx_val].cpu().numpy())
                    category_words.append(clean_token)
                    is_original.append(False)
                    
                    if neighbors_found == 10:
                        break
        
        if len(category_embeddings) == 0:
            print(f"No valid embeddings for {cat}")
            continue
            
        category_embeddings = np.array(category_embeddings)
        print(f"Plotting {len(category_embeddings)} words for {display_cat} using {model_name} Model...")
        
        n_samples = len(category_embeddings)
        if n_samples > 50:
            pca = PCA(n_components=50)
            reduced = pca.fit_transform(category_embeddings)
        else:
            reduced = category_embeddings
            
        perplexity = min(30, n_samples - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
        coords_2d = tsne.fit_transform(reduced)
        
        plt.figure(figsize=(16, 12))
        
        # Split original vs neighbor for plotting
        orig_indices = [i for i, val in enumerate(is_original) if val]
        neighbor_indices = [i for i, val in enumerate(is_original) if not val]
        
        # Plot neighbor words first (background)
        if neighbor_indices:
            sns.scatterplot(
                x=coords_2d[neighbor_indices, 0], 
                y=coords_2d[neighbor_indices, 1],
                color="blue",
                marker="o",
                s=100,
                alpha=0.4,
                edgecolor="w",
                label=f"Closest {model_name} Model Neighbors"
            )
            
        # Plot original words on top
        if orig_indices:
            sns.scatterplot(
                x=coords_2d[orig_indices, 0], 
                y=coords_2d[orig_indices, 1],
                color="red",
                marker="*",
                s=450,
                alpha=0.9,
                edgecolor="black",
                label="Original Influential Words"
            )
        
        # Add text labels
        for i, word in enumerate(category_words):
            if is_original[i]:
                plt.annotate(
                    word,
                    (coords_2d[i, 0], coords_2d[i, 1]),
                    xytext=(6, 6),
                    textcoords='offset points',
                    fontsize=12,
                    fontweight='bold',
                    color='darkred'
                )
            else:
                plt.annotate(
                    word,
                    (coords_2d[i, 0], coords_2d[i, 1]),
                    xytext=(4, 4),
                    textcoords='offset points',
                    fontsize=9,
                    alpha=0.7,
                    color='darkblue'
                )
                
        plt.title(f"{model_name} Model Space: {display_cat} (Influential Words & Their Top 10 Neighbors)", fontsize=16)
        plt.legend(loc='upper right', fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        filename = f"{model_name.lower()}_word_map_{cat.replace('avg_', '').replace('_drop', '')}.png"
        out_path = os.path.join(script_dir, filename)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Successfully saved {display_cat} plot to {filename}")

if __name__ == "__main__":
    main()
