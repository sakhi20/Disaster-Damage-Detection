"""
ResNet-18 Baseline Training Script — xBD Building Damage Detection
===================================================================
Trains a ResNet-18 classifier on a subset of the xBD dataset.
One sample per post-disaster tile; label = majority damage class in tile.

Usage:
    python scripts/train_baseline.py
    python scripts/train_baseline.py --data_dir data/xbd --epochs 10 \
        --train_subset 1000 --val_subset 200

Expected xBD directory structure:
    data/xbd/
    ├── train/
    │   ├── images/   *_pre_disaster.png  *_post_disaster.png
    │   └── labels/   *_post_disaster.json
    ├── hold/          (validation split)
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/

Extraction (run once before training):
    cat data/xview2_geotiff.tgz.part-* > data/xview2_geotiff.tgz
    tar -xzf data/xview2_geotiff.tgz -C data/
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# ─── Constants ────────────────────────────────────────────────────────────────
DAMAGE_CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
NUM_CLASSES = 4
LABEL_MAP = {name: i for i, name in enumerate(DAMAGE_CLASSES)}
IMG_SIZE = 224

# ImageNet normalization (ResNet pretrained stats)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─── Dataset ──────────────────────────────────────────────────────────────────
class XBDTileDataset(Dataset):
    """
    PyTorch Dataset for xBD post-disaster tiles.

    Each sample is one post-disaster PNG tile. The label is the majority
    damage class among all building polygons annotated in the tile's JSON.

    Args:
        samples:   List of (image_path: str, label: int) tuples.
        transform: torchvision transforms applied to each PIL image.
    """

    def __init__(self, samples: list, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ─── Data Collection ──────────────────────────────────────────────────────────
def collect_samples(data_dir: Path, split: str, max_samples: int) -> list:
    """
    Scan an xBD split directory and return (image_path, label) pairs.

    Args:
        data_dir:    Root of extracted xBD dataset (contains train/hold/test).
        split:       One of 'train', 'hold', 'test'.
        max_samples: Maximum number of tiles to load.

    Returns:
        List of (str, int) tuples — (post-disaster image path, damage label).

    Raises:
        FileNotFoundError: If the split directory does not exist.
    """
    images_dir = data_dir / split / "images"
    labels_dir = data_dir / split / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Directory not found: {images_dir}\n"
            f"  Ensure the xBD dataset has been extracted to: {data_dir}\n"
            f"  Extraction commands:\n"
            f"    cat data/xview2_geotiff.tgz.part-* > data/xview2_geotiff.tgz\n"
            f"    tar -xzf data/xview2_geotiff.tgz -C data/\n"
        )

    post_tiles = sorted(images_dir.glob("*_post_disaster.png"))
    if not post_tiles:
        raise FileNotFoundError(
            f"[ERROR] No post-disaster PNG files found in {images_dir}"
        )

    print(f"  Found {len(post_tiles)} post-disaster tiles in '{split}' split")

    samples = []
    skipped = 0

    for img_path in post_tiles[:max_samples]:
        stem = img_path.stem.replace("_post_disaster", "")
        label_path = labels_dir / f"{stem}_post_disaster.json"

        if not label_path.exists():
            skipped += 1
            continue

        label = _majority_label(label_path)
        samples.append((str(img_path), label))

    if skipped:
        print(f"  Skipped {skipped} tiles (missing label files)")

    return samples


def _majority_label(label_path: Path) -> int:
    """Parse xBD JSON and return the majority damage class index for the tile."""
    try:
        with open(label_path) as f:
            data = json.load(f)
        labels = [
            LABEL_MAP[feat["properties"]["subtype"]]
            for feat in data.get("features", {}).get("xy", [])
            if feat.get("properties", {}).get("subtype") in LABEL_MAP
        ]
        return int(np.bincount(labels).argmax()) if labels else 0
    except (json.JSONDecodeError, KeyError, ValueError):
        return 0  # default to no-damage on parse error


def _class_distribution(samples: list) -> str:
    counts = np.bincount([s[1] for s in samples], minlength=NUM_CLASSES)
    return "  ".join(f"{DAMAGE_CLASSES[i]}={counts[i]}" for i in range(NUM_CLASSES))


# ─── Model ────────────────────────────────────────────────────────────────────
def build_resnet18(num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    ResNet-18 pretrained on ImageNet-1K.
    Final FC layer replaced with Dropout + Linear for `num_classes` outputs.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


# ─── Training & Evaluation ────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

        if (batch_idx + 1) % 10 == 0:
            batch_acc = (preds == labels).float().mean().item()
            print(f"    Batch [{batch_idx+1:>3}/{len(loader)}]  "
                  f"loss={loss.item():.4f}  acc={batch_acc:.3f}")

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    per_class = {}
    for i, name in enumerate(DAMAGE_CLASSES):
        indices = [j for j, l in enumerate(all_labels) if l == i]
        if indices:
            per_class[name] = sum(all_preds[j] == i for j in indices) / len(indices)

    return total_loss / total, correct / total, per_class


# ─── Main ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="ResNet-18 baseline for xBD damage detection")
    p.add_argument("--data_dir",      type=str,   default="data/xbd")
    p.add_argument("--epochs",        type=int,   default=10)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--train_subset",  type=int,   default=1000,
                   help="Max training tiles (use -1 for full dataset)")
    p.add_argument("--val_subset",    type=int,   default=200,
                   help="Max validation tiles (use -1 for full dataset)")
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--save_path",     type=str,   default="results/baseline_resnet18.pth")
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    # ── Device ─────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 62)
    print("  ResNet-18 Baseline — xBD Building Damage Detection")
    print("=" * 62)
    print(f"  Device       : {device}" +
          (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"  Data dir     : {Path(args.data_dir).resolve()}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  LR           : {args.lr}")
    print(f"  Train subset : {args.train_subset if args.train_subset > 0 else 'full'}")
    print(f"  Val subset   : {args.val_subset if args.val_subset > 0 else 'full'}")
    print(f"  Save path    : {args.save_path}")
    print("=" * 62)

    # ── Transforms ─────────────────────────────────────────────────────────────
    train_transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # ── Data ───────────────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    max_train = args.train_subset if args.train_subset > 0 else 10**9
    max_val   = args.val_subset   if args.val_subset   > 0 else 10**9

    print("\n[INFO] Loading training samples...")
    train_samples = collect_samples(data_dir, "train", max_train)
    print(f"  → {len(train_samples)} samples  |  {_class_distribution(train_samples)}")

    print("[INFO] Loading validation samples...")
    val_samples = collect_samples(data_dir, "hold", max_val)
    print(f"  → {len(val_samples)} samples  |  {_class_distribution(val_samples)}")

    train_ds = XBDTileDataset(train_samples, transform=train_transform)
    val_ds   = XBDTileDataset(val_samples,   transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"))

    print(f"\n  Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    # ── Model ──────────────────────────────────────────────────────────────────
    print("\n[INFO] Building ResNet-18 (ImageNet pretrained)...")
    model = build_resnet18(NUM_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # ── Loss, Optimizer, Scheduler ─────────────────────────────────────────────
    # Upweight rare damage classes to counter xBD class imbalance
    class_weights = torch.tensor([0.5, 1.5, 2.0, 2.5], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── Training Loop ──────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  Starting Training")
    print("=" * 62)

    os.makedirs(Path(args.save_path).parent, exist_ok=True)
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        current_lr = scheduler.get_last_lr()[0]
        print(f"\n{'─' * 62}")
        print(f"  Epoch {epoch}/{args.epochs}   LR={current_lr:.2e}")
        print(f"{'─' * 62}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, per_class = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t_epoch
        print(f"\n  ✓ Epoch {epoch:2d}  [{elapsed:.0f}s]")
        print(f"    Train  loss={train_loss:.4f}  acc={train_acc:.4f} ({train_acc*100:.1f}%)")
        print(f"    Val    loss={val_loss:.4f}  acc={val_acc:.4f} ({val_acc*100:.1f}%)")
        print(f"    Per-class val accuracy:")
        for cls_name, acc in per_class.items():
            bar = "█" * int(acc * 20)
            print(f"      {cls_name:<15} {acc:.3f}  {bar}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "train_loss": train_loss,
                "history": history,
                "args": vars(args),
                "class_names": DAMAGE_CLASSES,
            }, args.save_path)
            print(f"    ★ New best val acc — checkpoint saved to {args.save_path}")

    # ── Final Summary ──────────────────────────────────────────────────────────
    total_min = (time.time() - t_start) / 60
    print("\n" + "=" * 62)
    print("  Training Complete")
    print("=" * 62)
    print(f"  Total time   : {total_min:.1f} min")
    print(f"  Best val acc : {best_val_acc:.4f} ({best_val_acc*100:.1f}%)")
    print(f"  Checkpoint   : {args.save_path}")
    print()
    print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*9}  {'─'*8}  {'─'*7}")
    for i in range(args.epochs):
        marker = " ★" if history["val_acc"][i] == best_val_acc else ""
        print(f"  {i+1:>5}  {history['train_loss'][i]:>10.4f}  "
              f"{history['train_acc'][i]:>9.4f}  "
              f"{history['val_loss'][i]:>8.4f}  "
              f"{history['val_acc'][i]:>7.4f}{marker}")
    print("=" * 62)


if __name__ == "__main__":
    main()
