import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score, hamming_loss, jaccard_score
from tqdm import tqdm

# --- 1. DATA SETUP ---

class EmotionDataset(Dataset):
    """
    Dataset for multi-label emotion classification.
    Handles tokenization and multi-hot encoding of 13 emotions.
    """
    def __init__(self, csv_path, tokenizer, max_length=128):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # The 13 target emotions as specified
        self.emotion_columns = [
            'Anger', 'Contempt', 'Disgust', 'Fear', 'Gratitude', 'Guilt', 
            'Happiness', 'Hope', 'Pride', 'Relief', 'Sadness', 'Sympathy', 
            'Emotions_Neutral'
        ]
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text = str(self.df.iloc[idx]['body'])
        labels = self.df.iloc[idx][self.emotion_columns].values.astype(float)
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float)
        }

def calculate_pos_weights(csv_path):
    """
    Calculates positive class weights (pos_weight) to handle imbalance.
    Formula: pos_weight = (num_negative_samples) / (num_positive_samples)
    """
    df = pd.read_csv(csv_path)
    emotion_columns = [
        'Anger', 'Contempt', 'Disgust', 'Fear', 'Gratitude', 'Guilt', 
        'Happiness', 'Hope', 'Pride', 'Relief', 'Sadness', 'Sympathy', 
        'Emotions_Neutral'
    ]
    
    labels = df[emotion_columns].values
    pos_counts = np.sum(labels, axis=0)
    neg_counts = len(df) - pos_counts
    
    # Avoid division by zero with small epsilon
    pos_weights = neg_counts / (pos_counts + 1e-6)
    return torch.tensor(pos_weights, dtype=torch.float)

# --- 2. MODEL INITIALIZATION ---

class RoBERTaEmotionClassifier(nn.Module):
    """
    RoBERTa-based multi-label classifier.
    Outputs raw logits for BCEWithLogitsLoss.
    """
    def __init__(self, n_classes=13):
        super(RoBERTaEmotionClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(0.1)
        # Custom head mapping 768 to 13 logits
        self.classifier = nn.Linear(self.roberta.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        # We use the pooled output ([CLS] token equivalent in RoBERTa)
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        return logits

# --- 3. TRAINING & EVALUATION LOOPS ---

def evaluate(model, data_loader, device, criterion):
    model.eval()
    losses = []
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            losses.append(loss.item())
            
            # Use 0.5 threshold for metrics
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
            targets = labels.cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(targets)
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    metrics = {
        'loss': np.mean(losses),
        'macro_f1': f1_score(all_targets, all_preds, average='macro'),
        'micro_f1': f1_score(all_targets, all_preds, average='micro'),
        'hamming_loss': hamming_loss(all_targets, all_preds),
        'macro_jaccard': jaccard_score(all_targets, all_preds, average='macro')
    }
    
    return metrics

def train():
    # Configuration
    TRAIN_CSV = 'Resources/UsVsThem_train_public.csv'
    VAL_CSV = 'Resources/UsVsThem_valid_public.csv'
    BATCH_SIZE = 16
    MAX_LEN = 128
    EPOCHS = 5
    LEARNING_RATE = 3e-5
    WEIGHT_DECAY = 0.01
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize Tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # Datasets & Loaders
    train_dataset = EmotionDataset(TRAIN_CSV, tokenizer, MAX_LEN)
    val_dataset = EmotionDataset(VAL_CSV, tokenizer, MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Handle Class Imbalance
    pos_weights = calculate_pos_weights(TRAIN_CSV).to(DEVICE)
    print(f"Calculated pos_weights: {pos_weights}")
    
    # Model, Optimizer, Loss
    model = RoBERTaEmotionClassifier(n_classes=13).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    
    # Scheduler
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    # Training State
    best_val_loss = float('inf')
    patience = 2
    trigger_times = 0
    
    for epoch in range(EPOCHS):
        model.train()
        train_losses = []
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            train_losses.append(loss.item())
            loop.set_description(f"Train Loss: {np.mean(train_losses):.4f}")
            
        # Evaluation
        val_metrics = evaluate(model, val_loader, DEVICE, criterion)
        print(f"Validation Results:")
        for k, v in val_metrics.items():
            print(f"  {k}: {v:.4f}")
            
        # Early Stopping
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            trigger_times = 0
            torch.save(model.state_dict(), 'best_roberta_model.pt')
            print("  Model saved!")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping triggered.")
                break

if __name__ == "__main__":
    # Ensure Resources directory exists or paths are correct before running
    # os.makedirs('Resources', exist_ok=True) 
    train()
