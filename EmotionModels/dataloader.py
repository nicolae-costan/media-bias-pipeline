# -*- coding: utf-8 -*-
import pandas as pd
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class RedditDataset(Dataset):
    """Reddit Emotion dataset."""

    EMOTION_COLUMNS = [
        'Anger', 'Contempt', 'Disgust', 'Fear', 'Gratitude',
        'Guilt', 'Happiness', 'Hope', 'Pride', 'Relief',
        'Sadness', 'Sympathy', 'Emotions_Neutral'
    ]

    def __init__(self, data_csv='file.csv'):
        """
        Args:
            data_csv (string): Path to the csv file with annotations.
        """
        self.comments = pd.read_csv(data_csv)

        # FIX: Store labels as a numpy float32 array instead of a list-of-lists
        # column on the DataFrame. This avoids per-sample pickle overhead when
        # num_workers > 0 and is faster to index.
        self.labels = self.comments[self.EMOTION_COLUMNS].values.astype(np.float32)

        # Keep only the text column in memory — drop unused columns
        self.texts = self.comments['body'].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # FIX: Return a plain dict instead of a pandas Series.
        # Pandas Series access adds overhead; plain dicts are faster and safer
        # when passed across DataLoader worker processes.
        return {
            'body': self.texts[idx],
            'label_aux': self.labels[idx],   # numpy array, shape [13]
        }


class MyCollator(object):
    def __init__(self, model_name, max_length):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def __call__(self, batch):
        output = {}

        # Clean URLs out of the text
        texts = [
            re.sub(
                r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*',
                'LINK',
                item['body'],
                flags=re.MULTILINE,
            )
            for item in batch
        ]

        # Tokenize the whole batch at once
        tokenized = self.tokenizer(
            texts,
            padding='longest',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
            add_special_tokens=True,
        )

        # FIX: Stack numpy arrays directly — much faster than building a list
        # of Python lists and letting torch.tensor() parse them.
        labels = np.stack([item['label_aux'] for item in batch])   # [B, 13]
        output['labels_aux'] = torch.from_numpy(labels)

        return tokenized.data, output


def sentiment_analysis_dataset(hparams, train=True, val=True, test=True):
    """
    Loads the Dataset from the csv files passed to the parser.
    """
    if train:
        return RedditDataset(hparams.train_csv)
    elif val:
        return RedditDataset(hparams.dev_csv)
    elif test:
        return RedditDataset(hparams.test_csv)

    return None