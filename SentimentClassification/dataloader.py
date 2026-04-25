# -*- coding: utf-8 -*-
import pandas as pd

from test_tube import HyperOptArgumentParser
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
import re, pickle, torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

# Emotions with enough training samples to learn generalizable patterns.
# Removed: Gratitude(33), Happiness(28), Relief(20), Guilt(81), Pride(93), Sadness(72)
# Kept: all classes with ≥177 positive training examples.
EMOTION_COLS = [
    'Anger',         # 931  (25.3%)
    'Contempt',      # 1396 (37.9%)
    'Disgust',       # 946  (25.7%)
    'Fear',          # 612  (16.6%)
    'Hope',          # 177  ( 4.8%)
    'Sympathy',      # 616  (16.7%)
    'Emotions_Neutral',  # 1120 (30.4%)
]

class RedditDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_csv = 'file.csv', aux_task = 'group', le = None, le_aux = None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.comments = pd.read_csv(data_csv)
        aux_task_str = str(aux_task)
        if aux_task_str == 'None':
            self.comments['label_aux'] = 0
        elif aux_task_str == 'emotions':
            # Use only EMOTION_COLS (7 learnable classes).
            # Rare classes like Relief(20), Happiness(28) are excluded;
            # they have too few examples for BERT to learn generalizable patterns.
            self.comments['label_aux'] = self.comments[EMOTION_COLS].values.tolist()
            self.columns = EMOTION_COLS
        else:
            # ONLY transform, don't re-fit! Fitting should happen in the model's __init__
            # on the full dataset to ensure consistent label indexing across splits.
            self.comments['label_aux'] = le_aux.transform(self.comments[aux_task_str].values)

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        return self.comments.iloc[idx][['body', 'label_aux', 'group', 'bias', 'usVSthem_scale']]


def pad_seq(seq, max_batch_len, pad_value):
    # IRL, use pad_sequence
    # https://pytorch.org/docs/master/generated/torch.nn.utils.rnn.pad_sequence.html
    return seq + (max_batch_len - len(seq)) * [pad_value]

class MyCollator(object):
    def __init__(self, model_name, max_length):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
    def __call__(self, batch):
        output = {}
        texts = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', 'LINK', comment['body'],
                        flags=re.MULTILINE) for comment in batch]
        tokenized = self.tokenizer(texts, padding='longest', truncation=True, max_length = self.max_length, return_tensors = 'pt', add_special_tokens = True)
        output['labels'] = torch.tensor([element['usVSthem_scale'] for element in batch], dtype=torch.float)
        output['labels_aux'] = torch.tensor([element['label_aux'] for element in batch], dtype=torch.float)
        return tokenized.data, output


def sentiment_analysis_dataset(
    hparams: HyperOptArgumentParser, train=True, val=True, test=True
):
    """
    Loads the Dataset from the csv files passed to the parser.
    :param hparams: HyperOptArgumentParser obj containg the path to the data files.
    :param train: flag to return the train set.
    :param val: flag to return the validation set.
    :param test: flag to return the test set.

    Returns:
        - Training Dataset, Development Dataset, Testing Dataset
    """
    if train:
        dataset = RedditDataset(hparams.train_csv, hparams.aux_task, hparams.le, hparams.le_aux)
    if val:
        dataset = RedditDataset(hparams.dev_csv, hparams.aux_task, hparams.le, hparams.le_aux)
    if test:
        dataset = RedditDataset(hparams.test_csv, hparams.aux_task, hparams.le, hparams.le_aux)
    return dataset
