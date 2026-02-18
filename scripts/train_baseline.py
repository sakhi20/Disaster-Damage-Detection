"""
Baseline ResNet-18 Training Script for xBD Building Damage Detection
=====================================================================
Trains a ResNet-18 classifier on a small subset of xBD data.
Target: ~1-2 hours on CPU, much faster on GPU.

Usage:
    python scripts/train_baseline.py
    python scripts/train_baseline.py --data_dir data/xbd --epochs 10 --subset 1000
    python scripts/train_baseline.py --demo   # runs with synthetic data (no xBD needed)

xBD Data Setup (if not yet extracted):
    cat data/xview2_geotiff.tgz.part-* > data/xview2_geotiff.tgz
    tar -xzf data/xview2_geotiff.tgz -C data/
    # Expected structure after extraction:
    # data/xbd/train/images/*.png
    # data/xbd/train/labels/*.json
    # data/xbd/hold/images/*.png
    # data/xbd/hold/labels/*.json
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


# ─── Dataset ──────────────────────────────────────────────────────────────────
class XBDPatchDataset(Dataset):
    """
    Loads individual building crops from xBD PNG images + JSON labels.
    Each JSON feature = one building polygon → one sample.
    For speed, we use the full tile image and the bounding box of each polygon.
    """

    def __init__(self, samples, transform=None):
        self.samples = samples   # list of (img_path, label_int)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class SyntheticXBDDataset(Dataset):
    """Synthetic dataset for demo/testing without real xBD data."""

    def __init__(self, n_samples=1000, transform=None):
        self.n = n_samples
        self.transform = transform
        # Simulate class imbalance similar to xBD
        weights = [0.55, 0.20, 0.15, 0.10]
        self.labels = np.random.choice(4, size=n_samples, p=weights).tolist()

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Random RGB image (simulates satellite patch)
        arr = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ─── Data Loading ─────────────────────────────────────────────────────────────
def collect_samples_from_xbd(data_dir: Path, split: str, max_samples: int):
    """
    Walk xBD split directory and collect (image_path, label) pairs.
    One sample per tile image (uses the majority damage class in that tile).
    """
    images_dir = data_dir / split / "images"
    labels_dir = data_dir / split / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(
            f"\n[ERROR] xBD images not found at: {images_dir}\n"
            f"  → Have you extracted the dataset? Run:\n"
            f"      cat data/xview2_geotiff.tgz.part-* > data/xview2_geotiff.tgz\n"
            f"      tar -xzf data/xview2_geotiff.tgz -C data/\n"
            f"  → Or run with --demo flag to use synthetic data:\n"
            f"      python scripts/train_baseline.py --demo\n"
        )

    samples = []
    post_images = sorted(images_dir.glob("*_post_disaster.png"))

    print(f"  Found {len(post_images)} post-disaster tiles in '{split}' split")

    for img_path in post_images[:max_samples]:
        stem = img_path.stem.replace("_post_disaster", "")
        label_path = labels_dir / f"{stem}_post_disaster.json"

        if not label_path.exists():
            continue

        # Parse JSON to get dominant damage class for this tile
        try:
            with open(label_path) as f:
                data = json.load(f)
            labels_in_tile = []
            for feat in data.get("features", {}).get("xy", []):
                subtype = feat.get("properties", {}).get("subtype", "no-damage")
                if subtype in LABEL_MAP:
                    labels_in_tile.append(LABEL_MAP[subtype])
            if not labels_in_tile:
                label_int = 0
            else:
                # Use majority class for the tile
                label_int = int(np.bincount(labels_in_tile).argmax())
        except Exception:
            label_int = 0

        samples.append((str(img_path), label_int))

    return samples


# ─── Model ────────────────────────────────────────────────────────────────────
def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """ResNet-18 pretrained on ImageNet, final FC replaced for damage classes."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


# ─── Training ─────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
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

        # Print batch progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            batch_acc = (preds == labels).float().mean().item()
            print(f"    Batch [{batch_idx+1}/{len(loader)}]  "
                  f"loss={loss.item():.4f}  batch_acc={batch_acc:.3f}")

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


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

    avg_loss = total_loss / total
    accuracy = correct / total

    # Per-class accuracy
    per_class = {}
    for cls_idx, cls_name in enumerate(DAMAGE_CLASSES):
        mask = [l == cls_idx for l in all_labels]
        if sum(mask) > 0:
            cls_correct = sum(p == l for p, l in zip(all_preds, all_labels) if l == cls_idx)
            per_class[cls_name] = cls_correct / sum(mask)

    return avg_loss, accuracy, per_class


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ResNet-18 baseline for xBD damage detection")
    parser.add_argument("--data_dir", type=str, default="data/xbd",
                        help="Path to extracted xBD dataset root")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train_subset", type=int, default=1000,
                        help="Max training tiles to use")
    parser.add_argument("--val_subset", type=int, default=200,
                        help="Max validation tiles to use")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--demo", action="store_true",
                        help="Use synthetic data (no xBD needed) for quick demo")
    parser.add_argument("--save_path", type=str, default="results/baseline_resnet18.pth")
    args = parser.parse_args()

    # ── Setup ──────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  ResNet-18 Baseline — xBD Building Damage Detection")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device      : {device}")
    if device.type == "cuda":
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  LR          : {args.lr}")
    print(f"  Train subset: {args.train_subset}")
    print(f"  Val subset  : {args.val_subset}")
    print(f"  Demo mode   : {args.demo}")
    print("=" * 60)

    # ── Transforms ─────────────────────────────────────────────────────────────
    train_transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    val_transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    # ── Dataset ────────────────────────────────────────────────────────────────
    if args.demo:
        print("\n[DEMO MODE] Using synthetic data (no real xBD needed)")
        train_ds = SyntheticXBDDataset(n_samples=args.train_subset, transform=train_transform)
        val_ds   = SyntheticXBDDataset(n_samples=args.val_subset,   transform=val_transform)
    else:
        data_dir = Path(args.data_dir)
        print(f"\n[INFO] Loading xBD data from: {data_dir.resolve()}")

        print("[INFO] Collecting training samples...")
        train_samples = collect_samples_from_xbd(data_dir, "train", args.train_subset)
        print(f"  → {len(train_samples)} training samples collected")

        print("[INFO] Collecting validation samples...")
        val_samples = collect_samples_from_xbd(data_dir, "hold", args.val_subset)
        print(f"  → {len(val_samples)} validation samples collected")

        # Print class distribution
        for split_name, samples in [("Train", train_samples), ("Val", val_samples)]:
            counts = np.bincount([s[1] for s in samples], minlength=4)
            dist = "  ".join(f"{DAMAGE_CLASSES[i]}={counts[i]}" for i in range(4))
            print(f"  {split_name} class dist: {dist}")

        train_ds = XBDPatchDataset(train_samples, transform=train_transform)
        val_ds   = XBDPatchDataset(val_samples,   transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"))

    print(f"\n[INFO] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Model ──────────────────────────────────────────────────────────────────
    print("\n[INFO] Building ResNet-18 model (ImageNet pretrained)...")
    model = build_model(NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params    : {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # ── Loss & Optimizer ───────────────────────────────────────────────────────
    # Class weights to handle xBD imbalance (more weight on rare damage classes)
    class_weights = torch.tensor([0.5, 1.5, 2.0, 2.5], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training Loop ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Starting Training")
    print("=" * 60)

    best_val_acc = 0.0
    os.makedirs(Path(args.save_path).parent, exist_ok=True)
    training_start = time.time()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        print(f"\n{'─'*60}")
        print(f"  Epoch {epoch}/{args.epochs}  (LR={scheduler.get_last_lr()[0]:.2e})")
        print(f"{'─'*60}")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        # Validate
        val_loss, val_acc, per_class_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step()
        epoch_time = time.time() - epoch_start

        # Log
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"\n  ✓ Epoch {epoch:2d} Summary  [{epoch_time:.0f}s]")
        print(f"    Train  →  loss={train_loss:.4f}  acc={train_acc:.4f} ({train_acc*100:.1f}%)")
        print(f"    Val    →  loss={val_loss:.4f}  acc={val_acc:.4f} ({val_acc*100:.1f}%)")
        print(f"    Per-class val accuracy:")
        for cls_name, acc in per_class_acc.items():
            bar = "█" * int(acc * 20)
            print(f"      {cls_name:<15} {acc:.3f}  {bar}")

        # Save best model
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
                "args": vars(args),
                "class_names": DAMAGE_CLASSES,
            }, args.save_path)
            print(f"    ★ New best! Saved to {args.save_path}")

    # ── Final Summary ──────────────────────────────────────────────────────────
    total_time = time.time() - training_start
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  Total time      : {total_time/60:.1f} minutes")
    print(f"  Best val acc    : {best_val_acc:.4f} ({best_val_acc*100:.1f}%)")
    print(f"  Model saved to  : {args.save_path}")
    print()
    print("  Epoch History:")
    print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*9}  {'─'*8}  {'─'*7}")
    for i in range(args.epochs):
        print(f"  {i+1:>5}  {history['train_loss'][i]:>10.4f}  "
              f"{history['train_acc'][i]:>9.4f}  "
              f"{history['val_loss'][i]:>8.4f}  "
              f"{history['val_acc'][i]:>7.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
