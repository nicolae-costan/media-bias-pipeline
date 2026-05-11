import torch
import torch.nn.functional as F
import numpy as np
import logging
from torch import nn
import pandas as pd

from torch_geometric.nn import GATConv
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

import pytorch_lightning as pl

log = logging.getLogger(__name__)

class GraphBiasLabels(pl.LightningModule):


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

    All architecture, loss, and optimizer hyperparameters are passed in via
    hparams (an argparse.Namespace)
    is declared once in add_model_specific_args() and flows through here.
    """

    def __init__(self, hparams):
        super().__init__()
        
        # Save clean hyperparameters to avoid TensorBoard issues (similar to BertRegression)
        if hasattr(hparams, '__dict__'):
            hparams_dict = vars(hparams)
        else:
            hparams_dict = dict(hparams)
            
        clean_hparams = {
            k: v for k, v in hparams_dict.items()
            if isinstance(v, (int, float, str, bool, type(None)))
        }
        self.save_hyperparameters(clean_hparams)
        
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

        # Step-output accumulators (PL 2.0 style)
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
        self.build_model()

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

    def build_model(self):
        import os
        try:
            if not os.path.exists(self.hparams.graph_path):
                log.info("Graph not found, building using build_graph.py...")
                os.system("python build_graph.py")
            self.graph_data = torch.load(self.hparams.graph_path)
            self.train_data = self.graph_data.train_mask
            self.val_data = self.graph_data.val_mask
            self.predict_data = self.graph_data.test_mask
        except Exception as e:
            log.error(f"Could not load or build graph: {e}")

    # -------------------------------------------------------------------------
    # Train / val / test steps
    # -------------------------------------------------------------------------

    def training_step(self, batch, batch_idx=0):
        data = batch
        logits = self(data.x, data.emotions, data.edge_index)
        loss = self.loss(logits, data.y, data.label_weights, data.train_mask)
        
        output = {"loss": loss}
        self.training_step_outputs.append(output)
        return output

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


    def on_train_epoch_end(self) -> None:
        outputs = self.training_step_outputs
        if outputs:
            train_loss_mean = torch.stack([o["loss"] for o in outputs]).mean()
            self.log("train_loss", train_loss_mean, prog_bar=True, sync_dist=True)
            self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx=0):
        data = batch
        output = self._evaluate(data, data.val_mask)
        self.validation_step_outputs.append(output)
        return output
        
    def on_validation_epoch_end(self) -> None:
        outputs = self.validation_step_outputs
        if outputs:
            val_loss_mean = torch.tensor([o["loss"] for o in outputs]).mean()
            val_acc_mean = torch.tensor([o["accuracy"] for o in outputs]).mean()
            val_f1_macro_mean = torch.tensor([o["f1_macro"] for o in outputs]).mean()
            val_f1_biased_mean = torch.tensor([o["f1_biased"] for o in outputs]).mean()
            
            self.log("val_loss", val_loss_mean, prog_bar=True, sync_dist=True)
            self.log("val_accuracy", val_acc_mean, prog_bar=True, sync_dist=True)
            self.log("val_f1_macro", val_f1_macro_mean, prog_bar=True, sync_dist=True)
            self.log("val_f1_biased", val_f1_biased_mean, prog_bar=True, sync_dist=True)
            self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx=0):
        data = batch
        output = self._evaluate(data, data.test_mask)
        self.test_step_outputs.append(output)
        return output
        
    def on_test_epoch_end(self) -> None:
        outputs = self.test_step_outputs
        if outputs:
            test_loss_mean = torch.tensor([o["loss"] for o in outputs]).mean()
            test_acc_mean = torch.tensor([o["accuracy"] for o in outputs]).mean()
            self.log("test_loss", test_loss_mean, prog_bar=True, sync_dist=True)
            self.log("test_accuracy", test_acc_mean, prog_bar=True, sync_dist=True)
            self.test_step_outputs.clear()
            
        model_save_path = "graph_model.pt"
        torch.save(self.state_dict(), model_save_path)
        log.info(f"Model saved to {model_save_path}")

    @torch.no_grad()
    def predict_unlabeled(self, output_csv="unlabeled_predictions.csv"):
        """
        Predicts labels for all articles in the graph that do not have a label (y == -1).
        This is perfect for inferring the 10k extra articles after training on the 4k.
        """
        self.eval()
        
        if not hasattr(self, 'graph_data') or self.graph_data is None:
            log.error("Graph data not loaded. Please call build_model() first.")
            return None
            
        data = self.graph_data
        # Move data to device if model is on GPU
        device = self.device
        x = data.x.to(device)
        emotions = data.emotions.to(device)
        edge_index = data.edge_index.to(device)
        
        logits = self(x, emotions, edge_index)
        preds = logits.argmax(dim=-1)
        
        # Unlabeled articles have y == -1
        unlabeled_mask = (data.y == -1)
        unlabeled_indices = unlabeled_mask.nonzero(as_tuple=True)[0].cpu().numpy()
        unlabeled_preds = preds[unlabeled_mask].cpu().numpy()
        

        records = []
        # Reverse map to convert 0/1 back to string labels
        label_map_inv = {0: "Non-biased", 1: "Biased"}
        
        for idx, pred in zip(unlabeled_indices, unlabeled_preds):
            # Graph data stores the original article IDs
            article_id = data.article_ids[idx]
            records.append({
                "article_id": article_id,
                "predicted_label": label_map_inv[pred]
            })
            
        df = pd.DataFrame(records)
        df.to_csv(output_csv, index=False)
        log.info(f"Saved predictions for {len(df)} unlabeled articles to {output_csv}")
        return df

    # -------------------------------------------------------------------------
    # Shared argparse args — declared ONCE, imported by both train.py & test.py
    # -------------------------------------------------------------------------


    @staticmethod
    def add_model_specific_args(parser):
        """
        Every model hyperparameter lives here.
        cls_input_dim / emo_input_dim are absent intentionally — read from the
        graph at runtime (see train_graph.py) to prevent shape mismatches.
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

        # Data
        parser.add_argument("--graph_path", type=str, default="graph.pt",
                            help="Path to the .pt file produced by build_graph.py")

        return parser