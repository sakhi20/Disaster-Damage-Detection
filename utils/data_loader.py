"""
xBD Dataset Loader for Building Damage Detection.

Loads pre/post disaster image pairs and building damage labels
from the xBD dataset structure.
"""

import os
import json
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
from rasterio.features import rasterize
import geopandas as gpd


# Damage class mapping
DAMAGE_CLASSES = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
    "un-classified": -1,
}

CLASS_NAMES = ["no-damage", "minor-damage", "major-damage", "destroyed"]


class XBDDataset(Dataset):
    """
    PyTorch Dataset for the xBD building damage detection dataset.

    Args:
        root_dir: Path to xBD dataset root (contains train/test/hold splits)
        split: One of 'train', 'test', 'hold'
        transform: Optional transforms applied to image tensors
        use_pre_post: If True, return concatenated pre+post images (6 channels)
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        use_pre_post: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.use_pre_post = use_pre_post

        self.samples = self._load_samples()

    def _load_samples(self) -> List[dict]:
        """Scan dataset directory and collect image/label pairs."""
        samples = []
        split_dir = self.root_dir / self.split
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"

        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        # Find all post-disaster images (labels are for post images)
        for img_path in sorted(images_dir.glob("*_post_disaster.png")):
            stem = img_path.stem.replace("_post_disaster", "")
            pre_path = images_dir / f"{stem}_pre_disaster.png"
            label_path = labels_dir / f"{stem}_post_disaster.json"

            if pre_path.exists() and label_path.exists():
                samples.append({
                    "pre": str(pre_path),
                    "post": str(img_path),
                    "label": str(label_path),
                    "name": stem,
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]

        # Load images
        pre_img = self._load_image(sample["pre"])
        post_img = self._load_image(sample["post"])

        # Load labels
        labels = self._load_labels(sample["label"])

        if self.use_pre_post:
            image = torch.cat([pre_img, post_img], dim=0)  # (6, H, W)
        else:
            image = post_img  # (3, H, W)

        if self.transform:
            image = self.transform(image)

        return image, labels

    def _load_image(self, path: str) -> torch.Tensor:
        """Load image as float tensor normalized to [0, 1]."""
        import cv2
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)

    def _load_labels(self, label_path: str) -> torch.Tensor:
        """Parse xBD JSON label file and return damage class tensor."""
        with open(label_path) as f:
            data = json.load(f)

        labels = []
        for feature in data.get("features", {}).get("xy", []):
            props = feature.get("properties", {})
            subtype = props.get("subtype", "no-damage")
            label = DAMAGE_CLASSES.get(subtype, 0)
            if label >= 0:
                labels.append(label)

        if not labels:
            return torch.tensor([], dtype=torch.long)
        return torch.tensor(labels, dtype=torch.long)


def get_dataloaders(
    root_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    transform_train=None,
    transform_val=None,
):
    """Create train/val/test DataLoaders for xBD dataset."""
    from torch.utils.data import DataLoader

    train_ds = XBDDataset(root_dir, split="train", transform=transform_train)
    val_ds = XBDDataset(root_dir, split="hold", transform=transform_val)
    test_ds = XBDDataset(root_dir, split="test", transform=transform_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
