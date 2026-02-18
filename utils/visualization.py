"""
Visualization utilities for building damage detection results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Optional, List


DAMAGE_COLORS = {
    0: (0.2, 0.8, 0.2, 0.7),   # no-damage: green
    1: (1.0, 1.0, 0.0, 0.7),   # minor-damage: yellow
    2: (1.0, 0.5, 0.0, 0.7),   # major-damage: orange
    3: (0.9, 0.1, 0.1, 0.7),   # destroyed: red
}

CLASS_NAMES = ["no-damage", "minor-damage", "major-damage", "destroyed"]


def visualize_predictions(
    pre_image: np.ndarray,
    post_image: np.ndarray,
    predictions: List[int],
    targets: Optional[List[int]] = None,
    title: str = "Building Damage Predictions",
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize pre/post image pair with predicted damage labels.

    Args:
        pre_image: Pre-disaster RGB image (H, W, 3)
        post_image: Post-disaster RGB image (H, W, 3)
        predictions: List of predicted damage class indices
        targets: Optional ground truth labels
        title: Plot title
        save_path: If provided, save figure to this path
    """
    ncols = 3 if targets is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    axes[0].imshow(pre_image)
    axes[0].set_title("Pre-Disaster")
    axes[0].axis("off")

    axes[1].imshow(post_image)
    axes[1].set_title("Post-Disaster + Predictions")
    axes[1].axis("off")

    if targets is not None:
        axes[2].imshow(post_image)
        axes[2].set_title("Ground Truth")
        axes[2].axis("off")

    # Legend
    patches = [
        mpatches.Patch(color=DAMAGE_COLORS[i][:3], label=CLASS_NAMES[i])
        for i in range(4)
    ]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> None:
    """Plot a normalized confusion matrix using seaborn."""
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_f1s: Optional[List[float]] = None,
    val_f1s: Optional[List[float]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot training and validation loss/F1 curves."""
    nrows = 2 if train_f1s is not None else 1
    fig, axes = plt.subplots(1, nrows, figsize=(7 * nrows, 5))
    if nrows == 1:
        axes = [axes]

    axes[0].plot(train_losses, label="Train Loss")
    axes[0].plot(val_losses, label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if train_f1s is not None:
        axes[1].plot(train_f1s, label="Train F1")
        axes[1].plot(val_f1s, label="Val F1")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Macro F1")
        axes[1].set_title("Training & Validation F1")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
