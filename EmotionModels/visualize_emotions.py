import chromadb
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.manifold import TSNE
from wordcloud import WordCloud
from tqdm import tqdm

# The 13 emotions defined in dataloader.py
EMOTIONS = [
    'Anger', 'Contempt', 'Disgust', 'Fear', 'Gratitude',
    'Guilt', 'Happiness', 'Hope', 'Pride', 'Relief',
    'Sadness', 'Sympathy', 'Neutral'
]

def generate_visualizations(
    db_path="./vector_db", 
    collection_name="article_embeddings", 
    input_csv="./data/merged_clean_data.csv",
    output_dir="./plots"
):
    print(f"--- Connecting to ChromaDB at {db_path} ---")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    client = chromadb.PersistentClient(path=db_path)
    
    try:
        collection = client.get_collection(name=collection_name)
        count = collection.count()
        print(f"Found {count} articles in DB. Fetching data...")

        # 1. Fetch Data from ChromaDB
        # We need embeddings for t-SNE and metadatas for emotions
        data = collection.get(include=['metadatas', 'embeddings', 'documents'])
        
        # Load into DataFrame
        df_emotions = pd.DataFrame([json.loads(m['emotion_scores']) for m in data['metadatas']], columns=EMOTIONS)
        df_emotions['article_id'] = data['ids']
        
        # 2. Join with CSV for Bias Labels
        if os.path.exists(input_csv):
            print(f"Joining with bias labels from {input_csv}...")
            df_csv = pd.read_csv(input_csv)[['article_id', 'label_bias']]
            df_csv['article_id'] = df_csv['article_id'].astype(str)
            df = pd.merge(df_emotions, df_csv, on='article_id', how='left')
        else:
            print("WARNING: input_csv not found. Bias-related plots will be skipped.")
            df = df_emotions
            df['label_bias'] = 'Unknown'

        # --- PLOT 1: EMOTION DISTRIBUTION ---
        print("Generating Emotion Distribution plot...")
        plt.figure(figsize=(12, 6))
        mean_scores = df[EMOTIONS].mean().sort_values(ascending=False)
        sns.barplot(x=mean_scores.index, y=mean_scores.values, palette="viridis")
        plt.title("Average Emotion Intensity across Dataset", fontsize=15)
        plt.ylabel("Average Probability", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "emotion_distribution.png"))
        plt.close()

        # --- PLOT 2: EMOTION vs BIAS HEATMAP ---
        if 'label_bias' in df.columns:
            print("Generating Bias-Emotion correlation plot...")
            bias_groups = df.groupby('label_bias')[EMOTIONS].mean()
            plt.figure(figsize=(12, 4))
            sns.heatmap(bias_groups, annot=True, cmap="YlOrRd", fmt=".2f")
            plt.title("Emotional Signature: Biased vs Non-Biased", fontsize=15)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "bias_emotion_heatmap.png"))
            plt.close()

        # --- PLOT 3: 2D GALAXY (t-SNE) ---
        # We use a sample of 2000 for speed
        print("Calculating 2D Projection (t-SNE) for 2000 samples...")
        sample_size = min(2000, len(data['embeddings']))
        indices = np.random.choice(len(data['embeddings']), sample_size, replace=False)
        
        sample_embeddings = np.array(data['embeddings'])[indices]
        sample_labels = df['label_bias'].iloc[indices]
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(sample_embeddings)
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], 
            hue=sample_labels, palette="Set1", alpha=0.7
        )
        plt.title("Article Galaxy (t-SNE): Clustering by Meaning & Bias", fontsize=15)
        plt.legend(title="Bias Label")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "article_galaxy_tsne.png"))
        plt.close()

        # --- PLOT 4: WORD CLOUDS ---
        print("Generating Word Clouds for top emotions...")
        # Create a word cloud for the top 3 most intense emotions
        top_emotions = mean_scores.index[:3]
        for emo in top_emotions:
            # Get text from articles where this emotion is high (> 0.5)
            high_emo_indices = df[df[emo] > 0.5].index
            if len(high_emo_indices) > 0:
                text_blob = " ".join([data['documents'][i] for i in high_emo_indices[:100]]) # Top 100 articles
                wc = WordCloud(width=800, height=400, background_color='white').generate(text_blob)
                plt.figure(figsize=(10, 5))
                plt.imshow(wc, interpolation='bilinear')
                plt.axis("off")
                plt.title(f"Key Words driving '{emo}'", fontsize=20)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"wordcloud_{emo.lower()}.png"))
                plt.close()

        print(f"\nSUCCESS! All plots saved to: {os.path.abspath(output_dir)}")
        print(f"- Distribution: emotion_distribution.png")
        print(f"- Bias Analysis: bias_emotion_heatmap.png")
        print(f"- 2D Galaxy: article_galaxy_tsne.png")
        print(f"- Word Clouds: wordcloud_*.png")

    except Exception as e:
        print(f"ERROR during visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_visualizations()
