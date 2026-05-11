import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ProgressBar
from pytorch_lightning import Trainer
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import jaccard_score

import warnings
warnings.filterwarnings("ignore")

EMOTION_COLS = [
    'Anger',         # GoEmotions idx 2 (anger)
    'Contempt',      # GoEmotions idx 10 (disapproval)
    'Disgust',       # GoEmotions idx 11 (disgust)
    'Fear',          # GoEmotions idx 14 (fear)
    'Hope',          # GoEmotions idx 20 (optimism)
    'Sympathy',      # GoEmotions idx 5 (caring)
    'Emotions_Neutral'# GoEmotions idx 27 (neutral)
]
GOEMOTIONS_MAPPING = [2, 10, 11, 14, 20, 5, 27]

class EmotionDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=128):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = self.data[EMOTION_COLS].values.astype(float)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['body'])
        labels = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float)
        }

class GoEmotionsFinetuner(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        # Load the fully trained 28-class model
        self.model = AutoModelForSequenceClassification.from_pretrained(self.hparams.model_name)
        
        # ---------------------------------------------------------
        # WEIGHT SURGERY: 28 classes -> 7 classes
        # ---------------------------------------------------------
        old_out_proj = self.model.classifier.out_proj
        in_features = old_out_proj.in_features
        out_features = len(GOEMOTIONS_MAPPING)
        
        new_out_proj = nn.Linear(in_features, out_features)
        
        # Copy over ONLY the weights for our 7 relevant labels
        with torch.no_grad():
            new_out_proj.weight.copy_(old_out_proj.weight[GOEMOTIONS_MAPPING, :])
            new_out_proj.bias.copy_(old_out_proj.bias[GOEMOTIONS_MAPPING])
            
        self.model.classifier.out_proj = new_out_proj
        self.model.num_labels = out_features
        self.model.config.num_labels = out_features
        
        self.threshold = 0.30
        
        # Freezing encoder for initial epochs isn't strictly necessary since we start 
        # from a very well-trained state, but we'll use a lower LR for the backbone.
        
        self.__build_loss()

    def __build_loss(self):
        train_df = pd.read_csv(self.hparams.train_csv)
        pos_counts = train_df[EMOTION_COLS].sum().values.astype(float)
        neg_counts = (len(train_df) - pos_counts).astype(float)
        
        pos_weight = torch.tensor(
            neg_counts / np.maximum(pos_counts, 1.0), dtype=torch.float32
        )
        print(f"Using pos_weight for BCE: {pos_weight.tolist()}")
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    def training_step(self, batch, batch_idx):
        logits = self(batch['input_ids'], batch['attention_mask'])
        loss = self.loss_fn(logits, batch['labels'])
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch['input_ids'], batch['attention_mask'])
        loss = self.loss_fn(logits, batch['labels'])
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return {'val_loss': loss, 'logits': logits, 'labels': batch['labels']}

    def validation_epoch_end(self, outputs):
        logits = torch.cat([x['logits'] for x in outputs])
        labels = torch.cat([x['labels'] for x in outputs])
        
        preds = (torch.sigmoid(logits) > self.threshold).long().cpu().numpy()
        targets = labels.long().cpu().numpy()
        
        acc = jaccard_score(targets, preds, average='macro', zero_division=0)
        self.log('val_acc', acc, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        # Differential learning rates:
        # Encoder (already highly trained) -> lower LR
        # Classifier (has our surgical weights, but adapting to new narrow dataset) -> slightly higher SLR
        optimizer_grouped_parameters = [
            {'params': self.model.roberta.parameters(), 'lr': self.hparams.learning_rate},
            {'params': self.model.classifier.parameters(), 'lr': self.hparams.learning_rate * 5}
        ]
        
        optimizer = torch.optim.Adam(optimizer_grouped_parameters)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2, eta_min=1e-7
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch", "frequency": 1}]

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    train_dataset = EmotionDataset(args.train_csv, tokenizer, max_length=args.max_length)
    val_dataset = EmotionDataset(args.dev_csv, tokenizer, max_length=args.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    model = GoEmotionsFinetuner(args)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath='lightning_logs/goemotions_finetune/',
        filename='model-{epoch:02d}-{val_acc:.3f}',
        save_top_k=1,
        verbose=True,
        monitor='val_acc',
        mode='max'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        patience=args.patience,
        verbose=True,
        mode='max'
    )
    
    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[checkpoint_callback, early_stop_callback],
        gradient_clip_val=1.0,
    )
    
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='SamLowe/roberta-base-go_emotions')
    parser.add_argument('--train_csv', type=str, default='Resources/UsVsThem_train_public.csv')
    parser.add_argument('--dev_csv', type=str, default='Resources/UsVsThem_valid_public.csv')
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--gpus', type=int, default=1)
    
    args = parser.parse_args()
    main(args)
