# -*- coding: utf-8 -*-
import pandas as pd
import re
import numpy as np
import torch
import os
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class RedditDataset(Dataset):
    """Reddit Emotion dataset."""

    EMOTION_COLUMNS = [
        'anger', 'disgust', 'fear', 'joy',
        'optimism', 'sadness', 'neutral'
    ]

    def __init__(self, data_csv='file.csv'):
        """
        Args:
            data_csv (string): Path to the csv file with annotations.
        """
        if not os.path.exists(data_csv):
            # Resolve relative to the script's own directory (handles running
            # from inside EmotionModels/ where "../data/..." is correct)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            alt_path = os.path.normpath(os.path.join(script_dir, '..', data_csv))
            if os.path.exists(alt_path):
                data_csv = alt_path
            else:
                # Fallback: prepend EmotionModels/ when run from project root
                alt_path2 = os.path.join("EmotionModels", data_csv)
                if os.path.exists(alt_path2):
                    data_csv = alt_path2

        self.comments = pd.read_csv(data_csv)

        # FIX: Store labels as a numpy float32 array instead of a list-of-lists
        # column on the DataFrame. This avoids per-sample pickle overhead when
        # num_workers > 0 and is faster to index.
        # Support both 'text' (combined dataset) and legacy 'body' (UsVsThem)
        text_col = 'text' if 'text' in self.comments.columns else 'body'

        # Drop rows where the text is NaN (some combined-dataset rows are empty)
        before = len(self.comments)
        self.comments = self.comments.dropna(subset=[text_col]).reset_index(drop=True)
        dropped = before - len(self.comments)
        if dropped:
            import warnings
            warnings.warn(f"Dropped {dropped} rows with NaN in '{text_col}' from {data_csv}")

        self.labels = self.comments[self.EMOTION_COLUMNS].values.astype(np.float32)

        # Keep only the text column in memory — drop unused columns
        self.texts = self.comments[text_col].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # FIX: Return a plain dict instead of a pandas Series.
        # Pandas Series access adds overhead; plain dicts are faster and safer
        # when passed across DataLoader worker processes.
        return {
            'body': self.texts[idx],
            'label_aux': self.labels[idx],   # numpy array, shape [num_emotions]
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
                str(item['body']),   # guard: cast to str in case of residual NaN
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
        labels = np.stack([item['label_aux'] for item in batch])   # [B, num_emotions]
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