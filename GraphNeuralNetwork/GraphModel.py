
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

    def forward(self, x, emotions, edge_index):
        cls_proj = self.cls_proj(x)
        emo_proj = self.emo_proj(emotions)

        h = self.pre_gat_linear(torch.cat([cls_proj, emo_proj], dim=-1))

        for gat, norm in zip(self.gat_layers, self.gat_norms):
            h = norm(F.dropout(gat(h, edge_index), p=self.dropout, training=self.training) + h)

        h = self.bottleneck(h)
        return self.classifier(h)

    def configure_optimizers(self):
        """
        Returns (optimizer, scheduler).

        Two param groups — different LRs because:
          - Projection MLPs are simple and converge fast → higher LR is fine
          - GAT attention weights are sensitive → lower LR avoids early collapse

        AdamW vs Adam:
            Adam folds weight decay into the adaptive moment update (wrong).
            AdamW decouples them, giving proper L2 regularization. Always prefer AdamW.

        ReduceLROnPlateau on val F1:
            Halves LR when val F1 stalls. Critical for GNNs which often need
            a smaller LR to push past an early plateau.
        """
        proj_params = (
                list(self.cls_proj.parameters()) +
                list(self.emo_proj.parameters()) +
                list(self.pre_gat_linear.parameters())
        )
        gat_params = (
                list(self.gat_layers.parameters()) +
                list(self.gat_norms.parameters()) +
                list(self.bottleneck.parameters()) +
                list(self.classifier.parameters())
        )

        optimizer = torch.optim.AdamW(
            [
                {"params": proj_params, "lr": self.lr_proj},
                {"params": gat_params, "lr": self.lr_gat},
            ],
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",  # maximize val F1
            patience=self.scheduler_patience,
            factor=self.scheduler_factor,
            min_lr=self.min_lr,
            verbose=True,
        )
        return optimizer, scheduler

    def train_step(self, data, optimizer):
        """
        One gradient update on the full graph.

        Returns:
            float: training loss value
        """
        self.train()
        optimizer.zero_grad()

        logits = self(data.x, data.emotions, data.edge_index)
        loss = self.loss(logits, data.y, data.label_weights, data.train_mask)
        loss.backward()

        # Clip gradients — GAT attention scores can spike early in training
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def validation_step(self, data):
        """
        Full-graph inference evaluated on val_mask nodes.

        Returns dict:
            loss      : float
            accuracy  : float
            f1_macro  : float  ← use this to drive the scheduler and early stopping
            f1_biased : float  ← F1 on the positive class (biased articles)
            report    : str    full sklearn classification report
        """
        self.eval()

        logits = self(data.x, data.emotions, data.edge_index)
        preds = logits.argmax(dim=-1)

        val_loss = self.loss(logits, data.y, data.label_weights, data.val_mask)

        valid = data.val_mask & (data.y != -1)
        y_true = data.y[valid].cpu().numpy()
        y_pred = preds[valid].cpu().numpy()

        acc = float((y_true == y_pred).mean())
        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        f1_bias = float(f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0))
        report = classification_report(
            y_true, y_pred,
            target_names=["Non-biased", "Biased"],
            zero_division=0,
        )

        return {
            "loss": val_loss.item(),
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_biased": f1_bias,
            "report": report,
        }