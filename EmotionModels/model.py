import argparse
import csv
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import jaccard_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from test_tube import HyperOptArgumentParser
import seaborn as sn
import matplotlib
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from transformers import get_linear_schedule_with_warmup, AutoModelForSequenceClassification, AutoModel, AutoTokenizer

from EmotionModels.dataloader import MyCollator


class EmotionModel(nn.Module):
    def __init__(self, hparams,model_id,extra_dropout):
        super().__init__()

        if hasattr(hparams,'__dict__'):
            hparams_dict = vars(hparams)
        else:
            hparams_dict = dict(hparams)

        clean_hparams = {

            k:v for k,v in hparams_dict.items()
            if isinstance(v,(int,float,str,bool,type(None)))
        }
        self.save_hyperparameters(clean_hparams)

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.batch_size = self.hparams.batch_size

        # 1. Initialize your custom collator
        self.prepare_sample = MyCollator(self.hparams.encoder_model, self.hparams.max_length)

        self.model = AutoModelForSequenceClassification.from_pretrained(self.hparams.encoder_model)

        # 3. The 13 -> 28 Expansion Mapping
        # 0:Anger, 1:Contempt, 2:Disgust, 3:Fear, 4:Gratitude, 5:Guilt, 6:Happiness,
        # 7:Hope, 8:Pride, 9:Relief, 10:Sadness, 11:Sympathy, 12:Neutral
        self.mapping = torch.tensor([
            6, 6, 0, 0, 12, 11, 12, 12, 7, 10, 1, 2, 5, 6, 3, 4,
            10, 6, 6, 3, 7, 8, 12, 9, 5, 10, 12, 12
        ], dtype=torch.long)

        # 4. Multi-label Loss Function
        self.loss_fn = nn.BCEWithLogitsLoss()

        model = AutoModelForSequenceClassification.from_pretrained(model_id)


    def forward(self,input_ids,attention_mask):
        outputs = self.model(input_ids=input_ids,attention_mask=attention_mask)
        return outputs.logits

    def calculate_los(self,logits_28,targets_13):

        mapping = self.mapping.to(targets_13.device)


model_id = "SamLowe/roberta-base-go_emotions"

# Instantiate your model (passing in a dummy value for extra_dropout)
model = EmotionModel(hparams=None, model_id=model_id, extra_dropout=0.1)