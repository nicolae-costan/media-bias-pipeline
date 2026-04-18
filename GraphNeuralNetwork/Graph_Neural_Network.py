import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pytorch_lightning as pl
from test_tube import HyperOptArgumentParser

class GraphNeuralNetwork(pl.LightningModule):


    def __init__(self, hparams: HyperOptArgumentParser):
        super().__init__()
        self.hparams = hparams



    @classmethod
    def add_model_specific_args(cls,parser:HyperOptArgumentParser):

        parser.opt_list(
            "--k_neighbors",
            default=5,
            type=int,
            options = [3,5,8],
            tunable = True,
            help = "Number of neigbors to connect in the graph"
        )
        # we add this so we don t have forced edges,
        parser.add_argument(
            "--similarity_threshold",
            default=0.80,
            type=float,
            help="Minimum cosine similarity required to create an edge."
        )

        parser.add_argument(
            "--gnn_num_layers",
            default=2,
            type=int

        )