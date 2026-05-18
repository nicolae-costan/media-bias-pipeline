import os
import sys
import pandas as pd
from tqdm import tqdm
import argparse
from collections import defaultdict

# Ensure we can import saliency_pipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saliency_pipeline import SaliencyPipeline
from backend.model_service import EMOTION_LABELS

def main():
    parser = argparse.ArgumentParser(description="Batch process BABE dataset for word-level bias and emotion saliency statistics.")
    parser.add_argument("--csv", type=str, default="data/final_labels_MBIC.csv", help="Path to BABE final labels CSV.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of articles to process for quick testing.")
    parser.add_argument("--output", type=str, default="WordImpactSaliency/babe_word_saliency_statistics.csv", help="Output path for aggregated stats.")
    args = parser.parse_args()

    # 1. Load BABE dataset
    if not os.path.exists(args.csv):
        print(f"Error: BABE dataset file not found at: {args.csv}")
        return

    print(f"Loading dataset from: {args.csv}...")
    df = pd.read_csv(args.csv, sep=";", on_bad_lines="skip")
    print(f"Successfully loaded {len(df):,} rows.")

    # Drop rows without valid text
    df = df.dropna(subset=["text"])
    articles = df["text"].tolist()

    if args.limit:
        print(f"Limiting execution to the first {args.limit} articles for verification.")
        articles = articles[:args.limit]

    # 2. Boot Pipeline
    print("Initializing Saliency Pipeline...")
    pipeline = SaliencyPipeline()

    # 3. Global Stats Accumulators
    # Structure: word -> { count, attention, bias_drop, anger_drop, disgust_drop, etc. }
    word_stats = defaultdict(lambda: {
        "count": 0,
        "attention": 0.0,
        "bias_drop": 0.0,
        **{f"{emo}_drop": 0.0 for emo in EMOTION_LABELS}
    })

    # Load NLTK English stopwords dynamically
    print("Loading NLTK English stopwords...")
    try:
        import nltk
        nltk.download('stopwords', quiet=True)
        from nltk.corpus import stopwords
        STOPWORDS = set(stopwords.words('english'))
    except ImportError:
        print("Warning: nltk library not found. Falling back to a standard English stopword list.")
        STOPWORDS = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", 
            "is", "are", "was", "were", "of", "to", "it", "that", "this", "these", "those",
            "they", "them", "their", "he", "she", "his", "her", "we", "our", "you", "your",
            "as", "by", "from", "at", "about", "has", "have", "had", "been", "would", "could", "should"
        }

    # 4. Processing Loop
    print(f"\nProcessing {len(articles)} articles...")
    processed_count = 0
    
    for text in tqdm(articles, desc="Processing Articles", unit="art"):
        try:
            # We ablate up to the top 5 attended words in each article
            res = pipeline.analyze_text_saliency(text, max_ablation_words=5)
            if not res or "word_saliency" not in res:
                continue

            for item in res["word_saliency"]:
                word = item["word"]
                
                # Filter out numbers, punctuation, short noise or structural stopwords
                if not word.isalpha() or len(word) <= 2 or word in STOPWORDS:
                    continue
                
                stats = word_stats[word]
                stats["count"] += 1
                stats["attention"] += item["attention"]
                stats["bias_drop"] += item["bias_drop"]
                
                for emo in EMOTION_LABELS:
                    stats[f"{emo}_drop"] += item["emotion_drops"].get(emo, 0.0)
                    
            processed_count += 1
        except Exception as e:
            # Silently handle rare subword offsets exceptions and keep batch processing robust
            continue

    print(f"\nCompleted processing. Successfully analyzed {processed_count}/{len(articles)} articles.")

    # 5. Calculate Averages & Compile Output DataFrame
    rows = []
    for word, stats in word_stats.items():
        count = stats["count"]
        # Only retain words that appear at least once
        if count > 0:
            row = {
                "word": word,
                "occurrences": count,
                "avg_attention": stats["attention"] / count,
                "avg_bias_drop": stats["bias_drop"] / count,
            }
            for emo in EMOTION_LABELS:
                row[f"avg_{emo}_drop"] = stats[f"{emo}_drop"] / count
            rows.append(row)

    if not rows:
        print("Warning: No words processed successfully. Check token boundaries.")
        return

    df_output = pd.DataFrame(rows)
    
    # Sort primarily by occurrences (frequency) then by bias impact drop
    df_output = df_output.sort_values(by=["occurrences", "avg_bias_drop"], ascending=[False, False])

    # 6. Save output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_output.to_csv(args.output, index=False)
    print(f"Global Word Saliency Database successfully saved to: {args.output}")
    print(f"Total unique words compiled: {len(df_output):,}")
    
    # Display top 10 words with both bias and emotion drops
    print("\n--- Top 10 Most Influential Words in Dataset ---")
    cols_to_display = ["word", "occurrences", "avg_attention", "avg_bias_drop"] + [f"avg_{emo}_drop" for emo in EMOTION_LABELS]
    # Set pandas display options so all columns print nicely without wrapping
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df_output.head(10)[cols_to_display])

if __name__ == "__main__":
    main()
