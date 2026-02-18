"""
Training script for building damage detection models.

Usage:
    python scripts/train.py --config configs/prithvi_finetune.yaml
    python scripts/train.py --config configs/vit_baseline.yaml --wandb
"""

import argparse
import os
import sys
from pathlib import Path

import yaml
import torch
import wandb

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import get_dataloaders
from utils.metrics import compute_metrics, print_classification_report


def parse_args():
    parser = argparse.ArgumentParser(description="Train damage detection model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"[INFO] Loaded config: {args.config}")
    print(f"[INFO] Model: {cfg['model']['name']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # W&B init
    if args.wandb and cfg["logging"].get("wandb", False):
        wandb.init(
            project=cfg["logging"]["project"],
            name=cfg["logging"]["run_name"],
            config=cfg,
        )

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(
        root_dir=cfg["data"]["root_dir"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    print(f"[INFO] Train batches: {len(train_loader)}")
    print(f"[INFO] Val batches: {len(val_loader)}")

    # TODO: Initialize model based on cfg["model"]["name"]
    # TODO: Training loop
    print("[INFO] Training script ready. Implement model and training loop.")


if __name__ == "__main__":
    main()
