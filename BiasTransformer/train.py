import argparse
import os

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from model import BiasTransformer


def run_stage(args, train_csv: str, dev_csv: str, test_csv: str, stage_name: str, init_checkpoint: str | None = None):
    args.train_csv = train_csv
    args.dev_csv = dev_csv
    args.test_csv = test_csv

    if init_checkpoint:
        model = BiasTransformer.load_from_checkpoint(init_checkpoint, hparams=args, strict=False)
    else:
        model = BiasTransformer(args)

    logger = TensorBoardLogger(save_dir=args.log_dir, name=stage_name)
    ckpt_dir = os.path.join(logger.log_dir, "checkpoints")
    checkpoint = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch}-{val_f1_macro:.4f}",
        monitor="val_f1_macro",
        mode="max",
        save_top_k=1,
    )
    early_stop = EarlyStopping(monitor="val_f1_macro", mode="max", patience=args.patience)

    trainer = Trainer(
        logger=logger,
        callbacks=[checkpoint, early_stop],
        accelerator="auto",
        devices=args.gpus,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.grad_clip,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
    )
    trainer.fit(model)
    trainer.test(model, ckpt_path="best")
    return checkpoint.best_model_path


def main():
    parser = argparse.ArgumentParser(description="Two-stage bias transformer training.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_dir", default="tb_logs")
    parser.add_argument("--data_dir", default="data/bias_transformer")
    parser.add_argument("--skip_pretrain", action="store_true")
    parser.add_argument("--pretrained_checkpoint", default=None)
    parser.add_argument("--limit_train_batches", default=1.0)
    parser.add_argument("--limit_val_batches", default=1.0)
    parser.add_argument("--limit_test_batches", default=1.0)
    parser = BiasTransformer.add_model_specific_args(parser)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    seed_everything(args.seed)

    init_checkpoint = args.pretrained_checkpoint
    if not args.skip_pretrain:
        pretrain_train = os.path.join(args.data_dir, "pretrain_train.csv")
        pretrain_valid = os.path.join(args.data_dir, "pretrain_valid.csv")
        pretrain_test = os.path.join(args.data_dir, "pretrain_test.csv")
        if os.path.exists(pretrain_train):
            init_checkpoint = run_stage(args, pretrain_train, pretrain_valid, pretrain_test, "bias_transformer_pretrain", init_checkpoint)
        else:
            print("[train] No pretrain split found; skipping graph pseudo-label stage.")

    run_stage(
        args,
        os.path.join(args.data_dir, "finetune_train.csv"),
        os.path.join(args.data_dir, "finetune_valid.csv"),
        os.path.join(args.data_dir, "finetune_test.csv"),
        "bias_transformer_finetune",
        init_checkpoint,
    )


if __name__ == "__main__":
    main()
