import argparse
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl


class BERTRegressor(pl.LightningModule):

    def __init__(self, hparams) -> None:
        super(BERTRegressor,self).__init__()
        if type(hparams) == dict:
            hparams = argparse.Namespace(**hparams)
        self.hparams = hparams
        self.batch_size = hparams.batch_size

        self.__build_model()
        self.__build_loss()
        if self.hparams.nr_frozen_epoch > 0:
            self.freeze_encoder()
        else:
            self._frozen = False

        self.nr_frozen_epochs = self.hparams.nr_frozen_epochs


    def __build_model(self):
        try:
            train_df = pd.read_csv(self.hparams.train_csv)
            test_df = pd.read_csv(self.hparams.test_csv)
            dev_df = pd.read_csv(self.hparams.dev_csv)
            comments = pd.concat([train_df, test_df, dev_df])
        except Exception as e:
            print(f"Could not load csv check for correct configurations path {e}")

        emotion_columns = [
            'Anger', 'Contempt', 'Disgust', 'Fear', 'Gratitude',
            'Guilt', 'Happiness', 'Hope', 'Pride', 'Relief',
            'Sadness', 'Sympathy', 'Emotions_Neutral'
        ]
        num_emotions = len(emotion_columns)