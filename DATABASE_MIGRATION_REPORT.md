# Project Update: Vector Database & Model Sync

This report summarizes the major architectural upgrades performed on the Media Bias Pipeline.

## 1. Database Migration: Postgres → ChromaDB (Vector DB)
We have successfully moved away from the PostgreSQL dependency. 

*   **New Database:** [ChromaDB](https://www.trychroma.com/) (stored locally in `/vector_db`).
*   **What is special about it?** 
    *   **Semantic Mapping:** It stores a 768-number "meaning map" (Embedding) for every article. You can now find articles that are "similar" in meaning, even if they use different words.
    *   **Emotional Metadata:** Each article also stores its 13-label probability scores. You can filter or search by "emotional intensity."
    *   **Fast Search:** It uses AI-optimized indexing to search 13k articles in milliseconds.
*   **Capacity:** 13,697 articles successfully embedded and stored.

## 2. Model Synchronization (The Retraining)
We performed a fresh training run for the Emotion Model.

*   **Why was this needed?** We updated the code to use a more robust "Mean Pooling" (MInT) architecture and 13 emotion labels. This change in the "blueprint" of the model meant that older checkpoints (saved with the old names) were no longer compatible with the new code.
*   **Result:** A fresh model was trained (Version 0 in root `tb_logs`) which is now perfectly synced with the latest inference logic.
*   **Future Note:** Retraining is generally NOT needed unless you change the underlying code architecture again.

## 3. Data Population Pipeline
We replaced the complex Spark-based script with a "Lite" (Pandas + Torch) version.

*   **Performance:** The new script processes the 13k articles in batches, calculating both the **768-dimensional meaning vector** and the **13-label emotion scores** simultaneously.
*   **Verification:** The `check_db.py` script confirms that articles are correctly stored with their IDs, text snippets, and emotion probabilities.

## 4. Commands Cheat Sheet

Use these commands to manage your pipeline:

### A. To Re-train the Model (Optional)
Run this if you ever change the `model.py` code:
```powershell
python EmotionModels/train.py --max_epochs 5 --batch_size 32
```

### B. To Process Articles (Populate the DB)
Run this to turn your CSV into AI vectors:
```powershell
python GraphNeuralNetwork/article_embeddings.py --input_csv "./data/merged_clean_data.csv" --model_checkpoint "tb_logs/emotion_classification/version_0/checkpoints/epoch=2-val_loss=0.0853.ckpt"
```

### C. To Verify the Database
Run this to see if the data is there:
```powershell
python GraphNeuralNetwork/check_db.py
```

### D. To Install Requirements
Run this if you move to a new computer:
```powershell
pip install -r EmotionModels/requirements.txt
```
