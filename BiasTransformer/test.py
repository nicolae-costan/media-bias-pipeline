import argparse
import os
import sys

import torch
from pytorch_lightning import Trainer, seed_everything

from model import BiasTransformer


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained bias transformer checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", type=int, default=1)
    parser = BiasTransformer.add_model_specific_args(parser)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    seed_everything(args.seed)
    model = BiasTransformer.load_from_checkpoint(args.checkpoint, hparams=args, strict=False)
    trainer = Trainer(accelerator="auto", devices=args.gpus)
    trainer.test(model)


if __name__ == "__main__":
    main()
