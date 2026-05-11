import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch_geometric.nn import GATConv
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

class GraphBiasLabels(nn.Module):
    """
    All architecture, loss, and optimizer hyperparameters are passed in via
    hparams (an argparse.Namespace)
    is declared once in add_model_specific_args() and flows through here.
    """

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        hp = hparams

        # --- Projection layers ---
        self.cls_proj = nn.Sequential(
            nn.Linear(hp.cls_input_dim, hp.embed_proj_dim),
            nn.ELU(),
        )
        self.emo_proj = nn.Sequential(
            nn.Linear(hp.emo_input_dim, hp.emo_proj_dim),
            nn.ELU(),
        )
        self.pre_gat_linear = nn.Sequential(
            nn.Linear(hp.embed_proj_dim + hp.emo_proj_dim, hp.hidden_dim),
            nn.ELU(),
        )

        # --- GAT layers + per-layer LayerNorm ---
        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()
        for _ in range(hp.number_gat_layers):
            self.gat_layers.append(
                GATConv(
                    in_channels=hp.hidden_dim,
                    out_channels=hp.hidden_dim // hp.gat_heads,
                    heads=hp.gat_heads,
                    concat=True,
                    dropout=hp.dropout,
                    add_self_loops=True,
                )
            )
            self.gat_norms.append(nn.LayerNorm(hp.hidden_dim))

        # --- Bottleneck + classifier ---
        self.bottleneck = nn.Sequential(
            nn.Linear(hp.hidden_dim, hp.bottleneck_dim),
            nn.LayerNorm(hp.bottleneck_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hp.bottleneck_dim, hp.num_classes)

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

    def forward(self, x, emotions, edge_index):
        h = self.pre_gat_linear(
            torch.cat([self.cls_proj(x), self.emo_proj(emotions)], dim=-1)
        )
        for gat, norm in zip(self.gat_layers, self.gat_norms):
            h = norm(
                F.dropout(gat(h, edge_index), p=self.hparams.dropout, training=self.training) + h
            )
        return self.classifier(self.bottleneck(h))

    # -------------------------------------------------------------------------
    # Loss
    # -------------------------------------------------------------------------

    def loss(self, logits, y, label_weights, mask):
        """
        Agreement-weighted cross-entropy with label smoothing.
        reduction='none' lets us multiply each sample's loss by its agreement
        score — F.cross_entropy's built-in `weight` arg is for class weights only.
        """
        valid = mask & (y != -1)
        logits_v = logits[valid]
        y_v = y[valid]
        weights_v = label_weights[valid]

        ce = F.cross_entropy(
            logits_v, y_v,
            reduction="none",
            label_smoothing=self.hparams.label_smoothing,
        )
        return (ce * weights_v).sum() / weights_v.sum().clamp(min=1e-6)

    # -------------------------------------------------------------------------
    # Optimizer + Scheduler
    # -------------------------------------------------------------------------

    def configure_optimizers(self):
        hp = self.hparams
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
                {"params": proj_params, "lr": hp.lr_proj},
                {"params": gat_params, "lr": hp.lr_gat},
            ],
            weight_decay=hp.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=hp.scheduler_patience,
            factor=hp.scheduler_factor,
            min_lr=hp.min_lr,
            verbose=True,
        )
        return optimizer, scheduler

    # -------------------------------------------------------------------------
    # Train / val / test steps
    # -------------------------------------------------------------------------

    def train_step(self, data, optimizer):
        self.train()
        optimizer.zero_grad()
        logits = self(data.x, data.emotions, data.edge_index)
        loss = self.loss(logits, data.y, data.label_weights, data.train_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def _evaluate(self, data, mask):
        """Shared evaluation logic for val and test masks."""
        self.eval()
        logits = self(data.x, data.emotions, data.edge_index)
        preds = logits.argmax(dim=-1)
        step_loss = self.loss(logits, data.y, data.label_weights, mask)

        valid = mask & (data.y != -1)
        y_true = data.y[valid].cpu().numpy()
        y_pred = preds[valid].cpu().numpy()

        return {
            "loss": step_loss.item(),
            "accuracy": float((y_true == y_pred).mean()),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_biased": float(f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
            "report": classification_report(
                y_true, y_pred,
                target_names=["Non-biased", "Biased"],
                zero_division=0,
            ),
        }

    def validation_step(self, data):
        return self._evaluate(data, data.val_mask)

    def test_step(self, data):
        return self._evaluate(data, data.test_mask)

    # -------------------------------------------------------------------------
    # Shared argparse args — declared ONCE, imported by both train.py & test.py
    # -------------------------------------------------------------------------

    @staticmethod
    def add_model_specific_args(parser):
        """
        Every model hyperparameter lives here. train.py and test.py both call
        this so the defaults are never repeated or duplicated across files.
        cls_input_dim / emo_input_dim are intentionally absent — they are read
        from the graph at runtime to prevent shape mismatches.
        """
        # Architecture
        parser.add_argument("--embed_proj_dim", type=int, default=256)
        parser.add_argument("--emo_proj_dim", type=int, default=64)
        parser.add_argument("--hidden_dim", type=int, default=128,
                            help="Must be divisible by --gat_heads")
        parser.add_argument("--bottleneck_dim", type=int, default=50)
        parser.add_argument("--number_gat_layers", type=int, default=2)
        parser.add_argument("--gat_heads", type=int, default=4)
        parser.add_argument("--num_classes", type=int, default=2)
        parser.add_argument("--dropout", type=float, default=0.2)

        # Loss
        parser.add_argument("--label_smoothing", type=float, default=0.05)

        # Optimizer
        parser.add_argument("--lr_proj", type=float, default=1e-3)
        parser.add_argument("--lr_gat", type=float, default=5e-4)
        parser.add_argument("--weight_decay", type=float, default=1e-4)

        # Scheduler
        parser.add_argument("--scheduler_patience", type=int, default=10)
        parser.add_argument("--scheduler_factor", type=float, default=0.5)
        parser.add_argument("--min_lr", type=float, default=1e-6)

        return parser