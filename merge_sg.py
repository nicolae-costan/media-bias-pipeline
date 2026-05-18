import os
from utils.utils import compute_agreement

def main():
    sg1_path = "data/raw_labels_SG1.csv"
    sg2_path = "data/raw_labels_SG2.csv"
    output_path = "data/consensus_labels_sg1_sg2.csv"

    if not os.path.exists(sg1_path) or not os.path.exists(sg2_path):
        print(f"Error: Could not find {sg1_path} or {sg2_path}")
        return

    print("Computing consensus...")
    df_consensus = compute_agreement(sg1_path, sg2_path, "article_id", "label_bias")
    
    df_consensus.to_csv(output_path, index=False)
    print(f"\nSuccess! Merged consensus saved to: {output_path}")

if __name__ == "__main__":
    main()
