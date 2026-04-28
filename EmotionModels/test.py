"""
Runs a trained model on the Test Set using a saved checkpoint.
"""
import os
from pathlib import Path

from model import EmotionModel
# FIX 1: Import the correct class from the correct file
from pytorch_lightning import Trainer, seed_everything
from test_tube import HyperOptArgumentParser


def main(hparams) -> None:
    """
    Main testing routine.
    """
    # FIX 2: Use Lightning's built-in seed tool
    seed_everything(hparams.seed)

    # ------------------------
    # 1. FIND & LOAD CHECKPOINT
    # ------------------------
    model = None
    # Look through the folder for the saved .ckpt file
    if os.path.exists(hparams.checkpoint_path):
        for file in os.listdir(hparams.checkpoint_path):
            if file.endswith(".ckpt"):
                ckpt_full_path = os.path.join(hparams.checkpoint_path, file)
                print(f"--- Loading model from: {ckpt_full_path} ---")
                model = EmotionModel.load_from_checkpoint(ckpt_full_path, hparams=hparams)
                break

    if model is None:
        raise FileNotFoundError(f"Could not find a .ckpt file in {hparams.checkpoint_path}")

    # Ensure the model knows where to save its 'predictions.csv'
    model.hparams.checkpoint_path = hparams.checkpoint_path

    # ------------------------
    # 2. INIT TRAINER
    # ------------------------
    # We don't need a logger for testing, so we set it to False
    trainer = Trainer(
        logger=False,
        accelerator="cpu",  # Force CPU usage
        devices=1,  # Use 1 CPU core/process
    )
    # ------------------------
    # 3. START TESTING
    # ------------------------
    # This calls the test_step and test_epoch_end functions in your model
    trainer.test(model)


if __name__ == "__main__":
    parser = HyperOptArgumentParser(
        strategy="random_search",
        description="BERT Multi-task Tester",
        add_help=True,
    )

    # Path to the FOLDER where your .ckpt and logs are stored
    parser.add_argument(
        "--checkpoint_path",
        default="tb_logs/task_emotions/version_0",
        type=str,
        help="Path to the directory containing the .ckpt file"
    )

    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--batch_size", default=6, type=int)

    # Re-add model-specific args so the loading logic understands the hparams
    parser = EmotionModel.add_model_specific_args(parser)
    hparams = parser.parse_args()

    main(hparams)