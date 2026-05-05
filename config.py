"""
Central configuration for machine-dependent paths used in assignment scripts.

"""

from pathlib import Path

# To find the location of the project folder moves
ROOT_DIR = Path(__file__).resolve().parent

# Path to extracted CIFAR-10 python files (directory containing data_batch_1 ... test_batch)
DATA_DIR = ROOT_DIR / "datasets"/ "cifar-10-batches-py"

# Optional logging directory (for wandb/tensorboard/custom logs)
LOG_DIR = ROOT_DIR / "logs"

# Directory where trained model checkpoints are stored
MODEL_SAVE_DIR = ROOT_DIR / "saved_models"

# Create folders automatically if they do not exist.
LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
