"""
Evaluation script for building damage detection models.

Usage:
    python scripts/evaluate.py --config configs/prithvi_finetune.yaml \
        --checkpoint checkpoints/prithvi_finetune/best_model.pth
"""

import argparse
import sys
from pathlib import Path

import yaml
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import get_dataloaders
from utils.metrics import compute_metrics, print_classification_report, get_confusion_matrix
from utils.visualization import plot_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate damage detection model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["test", "hold"])
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Evaluating on {args.split} split")
    print(f"[INFO] Checkpoint: {args.checkpoint}")

    # TODO: Load model and run evaluation
    print("[INFO] Evaluation script ready. Implement model loading and inference.")


if __name__ == "__main__":
    main()
