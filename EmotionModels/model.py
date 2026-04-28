import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from transformers import get_linear_schedule_with_warmup, AutoModelForSequenceClassification
from sklearn.metrics import jaccard_score
import logging

# Import your custom dataset and collator logic
from dataloader import sentiment_analysis_dataset, MyCollator

log = logging.getLogger(__name__)


class EmotionModel(pl.LightningModule):
    def __init__(self, hparams) -> None:
        super(EmotionModel, self).__init__()

        # 1. Clean hparams to prevent TensorBoard/Logger crashes
        if hasattr(hparams, '__dict__'):
            hparams_dict = vars(hparams)
        else:
            hparams_dict = dict(hparams)

        clean_hparams = {
            k: v for k, v in hparams_dict.items()
            if isinstance(v, (int, float, str, bool, type(None)))
        }
        self.save_hyperparameters(clean_hparams)

        # 2. State management for Lightning 2.0+
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # 3. Model & Collator Setup
        self.prepare_sample = MyCollator(self.hparams.encoder_model, self.hparams.max_length)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.hparams.encoder_model)

        # 4. Label Expansion Mapping (13 Custom Labels -> 28 GoEmotions Nodes)
        # Maps indices: Anger(0), Contempt(1), Disgust(2), Fear(3), Gratitude(4), Guilt(5),
        # Happiness(6), Hope(7), Pride(8), Relief(9), Sadness(10), Sympathy(11), Neutral(12)
        self.mapping = torch.tensor([
            6, 6, 0, 0, 12, 11, 12, 12, 7, 10, 1, 2, 5, 6, 3, 4,
            10, 6, 6, 3, 7, 8, 12, 9, 5, 10, 12, 12
        ], dtype=torch.long)

        self.loss_fn = nn.BCEWithLogitsLoss()

    def setup(self, stage: str = None):
        """
        Called before training/validation/testing begins.
        Putting the model into train mode here ensures all submodules
        are in train mode before Lightning inspects them.
        """
        if stage == "fit":
            self.model.train()
            for module in self.model.modules():
                module.train()

    def on_train_start(self):
        """
        Belt-and-suspenders: also force train mode right before the first step.
        This covers any Lightning internals that might flip modules back to eval
        between setup() and the first training_step().
        """
        self.model.train()
        for module in self.model.modules():
            module.train()

    def _safe_squeeze(self, inputs):
        """
        Checks tensor dimensions to safely remove extra dimensions added by dataloaders
        without collapsing valid single-word sequences.
        """
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        if input_ids.dim() == 3 and input_ids.size(1) == 1:
            input_ids = input_ids.squeeze(1)
        if attention_mask.dim() == 3 and attention_mask.size(1) == 1:
            attention_mask = attention_mask.squeeze(1)

        return input_ids, attention_mask

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    def calculate_loss(self, logits_28, targets_13):
        # Move mapping to same device as targets
        mapping = self.mapping.to(targets_13.device)
        targets_28 = targets_13[:, mapping]
        loss = self.loss_fn(logits_28, targets_28)
        return loss, targets_28

    def training_step(self, batch, batch_nb):
        inputs, targets = batch
        input_ids, attention_mask = self._safe_squeeze(inputs)

        logits_28 = self.forward(input_ids, attention_mask)
        loss, _ = self.calculate_loss(logits_28, targets['labels_aux'])

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.training_step_outputs.append({"loss": loss})
        return loss

    def validation_step(self, batch, batch_nb):
        inputs, targets = batch
        input_ids, attention_mask = self._safe_squeeze(inputs)

        logits_28 = self.forward(input_ids, attention_mask)
        loss, targets_28 = self.calculate_loss(logits_28, targets['labels_aux'])

        # Convert logits to binary predictions (0 or 1)
        preds_28 = (logits_28 > 0).float()

        # We store the TENSORS. Lightning handles moving them to CPU/GPU automatically.
        output = {
            "val_loss": loss,
            "preds": preds_28,
            "targets": targets_28
        }
        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        # 1. Calculate Average Loss (Averaging loss is standard practice)
        avg_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()

        # 2. Concatenate all predictions and targets from all batches
        # This creates two large matrices of shape [Total_Samples, 28]
        all_preds = torch.cat([x["preds"] for x in self.validation_step_outputs], dim=0)
        all_targets = torch.cat([x["targets"] for x in self.validation_step_outputs], dim=0)

        # 3. Move to CPU and convert to Numpy for Scikit-Learn
        all_preds_np = all_preds.cpu().numpy()
        all_targets_np = all_targets.cpu().numpy()

        # 4. Calculate the TRUE Global Jaccard Score
        global_jaccard = jaccard_score(
            all_targets_np,
            all_preds_np,
            average="macro",
            zero_division=0
        )

        # 5. Log the results
        self.log("val_loss", avg_loss, prog_bar=True, sync_dist=True)
        self.log("val_jaccard", global_jaccard, prog_bar=True, sync_dist=True)

        # 6. Clear memory for the next epoch
        self.validation_step_outputs.clear()
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.encoder_learning_rate)

        # Calculate total steps for the scheduler
        train_loader = self.train_dataloader()
        train_steps = len(train_loader) * self.hparams.max_epochs
        warmup_steps = int(self.hparams.warmup_proportion * train_steps)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=train_steps
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    # Dataloader Methods
    def train_dataloader(self):
        dataset = sentiment_analysis_dataset(self.hparams, train=True, val=False, test=False)
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True,
                          collate_fn=self.prepare_sample, num_workers=self.hparams.loader_workers)

    def val_dataloader(self):
        dataset = sentiment_analysis_dataset(self.hparams, train=False, val=True, test=False)
        return DataLoader(dataset, batch_size=self.hparams.batch_size,
                          collate_fn=self.prepare_sample, num_workers=self.hparams.loader_workers)

    @classmethod
    def add_model_specific_args(cls, parser):
        parser.add_argument("--encoder_model", default="SamLowe/roberta-base-go_emotions", type=str)
        parser.add_argument("--encoder_learning_rate", default=2e-5, type=float)
        parser.add_argument("--warmup_proportion", default=0.1, type=float)
        parser.add_argument("--max_length", default=128, type=int)
        parser.add_argument("--loader_workers", default=0, type=int)  # Set to 0 for Windows stability
        parser.add_argument("--train_csv", default="Resources/UsVsThem_train_public.csv", type=str)
        parser.add_argument("--dev_csv", default="Resources/UsVsThem_valid_public.csv", type=str)
        parser.add_argument("--test_csv", default="Resources/UsVsThem_test_public.csv", type=str)
        return parser