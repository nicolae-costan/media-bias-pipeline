
import os
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops
from test_tube import HyperOptArgumentParser
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
from torch_geometric.nn import GATConv
"""
Architecture:
    Input features [N, 781]  (768 CLS + 13 emotions)
        → Embeddings Projection                      -> Sentiment Projection
        
          [N, 768]                                    [N,13]
        
        -> Linear Proj [N,embeding_dim]            -> Linear Proj [N,emotion_din]
                                            |
                                            |
                                            |
                                            ^
                                -> Concat layers [N,embedding_dim+emotion_dim]
                                -> Concat input game Linear [N,dim]->N,hidden_dim]
                                → GAT layer 1        [N, hidden_dim]
                                → GAT layer 2        [N, hidden_dim]
                                → Linear bottleneck  [N, 50]          ← the 50-dim latent space
                                → Classification head [N, 2]
"""


class GraphBiasLabels(nn.Module):
    def __init__(
            self,
            cls_input_dim: int = 768,
            emo_input_dim: int = 13,
            embed_proj_dim: int = 256,
            emo_proj_dim: int = 64,
            hidden_dim: int = 128,
            bottleneck_dim: int = 50,
            dropout: float = 0.2,
            number_gat_layers: int = 2,
            num_classes: int = 2,
            gat_heads: int = 4,


    ):
        super(GraphBiasLabels, self).__init__()
        self.dropout = dropout

        # 1. Embeddings Projection
        self.cls_proj = nn.Sequential(
            nn.Linear(cls_input_dim, embed_proj_dim),
            nn.ELU(),

        )

        self.emo_proj = nn.Sequential(
            nn.Linear(emo_input_dim, emo_proj_dim),
            nn.ELU(),

        )

        combined_dim = embed_proj_dim + emo_proj_dim
        self.pre_gat_linear = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ELU(),
        )

        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()

        for _ in range(number_gat_layers):
            # concat=True means output is num_heads * (hidden_dim // num_heads) = hidden_dim
            self.gat_layers.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // gat_heads,
                    heads=gat_heads,
                    concat=True,
                    dropout=dropout,
                    add_self_loops=True,
                )
            )
            self.gat_norms.append(nn.LayerNorm(hidden_dim))

        # Bottleneck: compress to 50 dims
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.ReLU(),
        )

        # Classification head
        self.classifier = nn.Linear(bottleneck_dim, num_classes)


    def loss(self,logits,y,label_weights,mask):
        '''
        y is the value there might be unlabeled data
        label_weights are the aggrement score per article
        mask is the data that we train on
        '''
        loss = nn.CrossEntropyLoss()

        valid = mask & (y != -1)
        logits_m = logits[valid]
        y_m = y[valid]
        weights = label_weights[valid]

        ce = F.cross_entropy(logits_m, y_m, weights)

        loss = (ce * weights).sum() / weights.sum().clamp(min=1e-6)
        return loss

