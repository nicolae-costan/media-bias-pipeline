# -*- coding: utf-8 -*-
import pandas as pd
import re
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class RedditDataset(Dataset):
    """Reddit Emotion dataset."""

    # Removed aux_task, le, and le_aux from the initialization
    def __init__(self, data_csv='file.csv'):
        """
        Args:
            data_csv (string): Path to the csv file with annotations.
        """
        self.comments = pd.read_csv(data_csv)

        self.columns = [
            'Anger', 'Contempt', 'Disgust', 'Fear', 'Gratitude',
            'Guilt', 'Happiness', 'Hope', 'Pride', 'Relief',
            'Sadness', 'Sympathy', 'Emotions_Neutral'
        ]

        # Group the 13 emotion columns into a single list per row
        self.comments['label_aux'] = self.comments[self.columns].values.tolist()

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        # We only need the text and the 13 emotions now!
        return self.comments.iloc[idx][['body', 'label_aux']]


class MyCollator(object):
    def __init__(self, model_name, max_length):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def __call__(self, batch):
        output = {}
        # Clean URLs out of the text
        texts = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', 'LINK', comment['body'],
                        flags=re.MULTILINE) for comment in batch]

        # Tokenize
        tokenized = self.tokenizer(
            texts,
            padding='longest',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
            add_special_tokens=True
        )

        # Convert targets to a PyTorch tensor
        output['labels_aux'] = torch.tensor([element['label_aux'] for element in batch], dtype=torch.float)

        return tokenized.data, output


def sentiment_analysis_dataset(hparams, train=True, val=True, test=True):
    """
    Loads the Dataset from the csv files passed to the parser.
    """
    dataset = None

    # Removed hparams.aux_task, hparams.le, and hparams.le_aux
    if train:
        dataset = RedditDataset(hparams.train_csv)
    elif val:
        dataset = RedditDataset(hparams.dev_csv)
    elif test:
        dataset = RedditDataset(hparams.test_csv)

    return dataset