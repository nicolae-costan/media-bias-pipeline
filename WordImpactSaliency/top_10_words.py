import pandas as pd
import sys

csv_path = r"C:\Users\rares\Desktop\BigData\media-bias-pipeline\WordImpactSaliency\babe_word_saliency_statistics.csv"
df = pd.read_csv(csv_path)

categories = [
    "avg_bias_drop", "avg_anger_drop", "avg_disgust_drop", 
    "avg_fear_drop", "avg_joy_drop", "avg_optimism_drop", 
    "avg_sadness_drop", "avg_neutral_drop"
]

# Filter for words that have a reasonable frequency (>= 3)
df_filtered = df[df["occurrences"] >= 3].copy()

for cat in categories:
    print(f"\n## Top 10 Most Influential Words for {cat.replace('avg_', '').replace('_drop', '').title()}")
    # Create a temporary column for absolute values
    df_filtered['abs_' + cat] = df_filtered[cat].abs()
    top10 = df_filtered.sort_values(by='abs_' + cat, ascending=False).head(10)
    for i, row in top10.iterrows():
        print(f"- **{row['word']}**: {row[cat]:.4f} drop (occurrences: {row['occurrences']})")