import argparse
import re
import os
import sys
import time
import json
import pandas as pd
import numpy as np
import torch
import chromadb
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Caches for model and tokenizer
_model_cache = {}
_tokenizer_cache = {}

def get_args():
    parser = argparse.ArgumentParser(description='Embed articles using EmotionModel (Lite Version)')

    parser.add_argument("--input_csv", type=str, default="./data/merged_clean_data.csv")
    parser.add_argument("--text_column", type=str, default="body")
    parser.add_argument("--id_column", type=str, default="article_id")

    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--num_groups", type=int, default=13)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16) # Reduced for safety

    parser.add_argument("--db_path", type=str, default="./vector_db")
    parser.add_argument("--collection_name", type=str, default="article_embeddings")

    return parser.parse_args()

def load_model_and_tokenizer(checkpoint_path):
    global _model_cache, _tokenizer_cache
    
    if checkpoint_path not in _model_cache:
        print(f"--- Loading model from {checkpoint_path} ---")
        
        # Add paths to find EmotionModels.model
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        emotion_dir = os.path.join(parent_dir, 'EmotionModels')
        if parent_dir not in sys.path: sys.path.append(parent_dir)
        if emotion_dir not in sys.path: sys.path.append(emotion_dir)

        from EmotionModels.model import EmotionModel
        from transformers import AutoTokenizer

        # Load model
        model = EmotionModel.load_from_checkpoint(checkpoint_path, map_location="cpu")
        model.eval()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model.hparams.encoder_model)
        
        _model_cache[checkpoint_path] = model
        _tokenizer_cache[checkpoint_path] = tokenizer
        
    return _model_cache[checkpoint_path], _tokenizer_cache[checkpoint_path]

def _clean_text(text: str) -> str:
    if not isinstance(text, str): return ""
    return re.sub(
        r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*',
        'LINK', text, flags=re.MULTILINE
    )

def main():
    args = get_args()
    
    if not os.path.exists(args.input_csv):
        print(f"ERROR: File not found at {args.input_csv}")
        return

    # 1. Load Data
    print(f"--- Reading {args.input_csv} ---")
    df = pd.read_csv(args.input_csv).dropna(subset=[args.id_column, args.text_column])
    total_articles = len(df)
    print(f"Found {total_articles} valid articles.")

    # 2. Load Model
    model, tokenizer = load_model_and_tokenizer(args.model_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Running on: {device}")

    # 3. Setup ChromaDB
    print(f"--- Initializing Vector DB at {args.db_path} ---")
    client = chromadb.PersistentClient(path=args.db_path)
    collection = client.get_or_create_collection(name=args.collection_name, metadata={"hnsw:space": "cosine"})

    # 4. Processing Loop
    print(f"--- Starting Embedding Process (Batch Size: {args.batch_size}) ---")
    
    for start_idx in tqdm(range(0, total_articles, args.batch_size)):
        end_idx = min(start_idx + args.batch_size, total_articles)
        batch_df = df.iloc[start_idx:end_idx]
        
        ids = batch_df[args.id_column].astype(str).tolist()
        texts = [_clean_text(t) for t in batch_df[args.text_column].tolist()]
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            # Get embeddings via Mean Pooling (MInT)
            outputs = model.model(
                input_ids=tokenized['input_ids'],
                attention_mask=tokenized['attention_mask'],
                output_hidden_states=True
            )
            
            last_hidden = outputs.hidden_states[-1] 
            mask = tokenized['attention_mask'].unsqueeze(-1)
            sum_hidden = (last_hidden * mask).sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1e-9)
            embeddings = (sum_hidden / count).cpu().numpy()
            
            # Get Emotion Scores
            logits_aux = model.classifier(sum_hidden / count)
            emotions = torch.sigmoid(logits_aux).cpu().numpy()

        # Prepare for ChromaDB
        final_ids = ids
        final_embeddings = embeddings.tolist()
        final_metadatas = []
        
        for i in range(len(ids)):
            final_metadatas.append({
                "emotion_scores": json.dumps(emotions[i].tolist()),
                "text_snippet": texts[i][:200]
            })

        # Save
        collection.upsert(
            ids=final_ids,
            embeddings=final_embeddings,
            metadatas=final_metadatas,
            documents=texts
        )

    print(f"\n--- SUCCESS! {total_articles} articles processed and stored in {args.db_path} ---")

if __name__ == "__main__":
    main()