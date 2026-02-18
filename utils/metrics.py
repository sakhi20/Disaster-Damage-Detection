"""
Evaluation metrics for building damage detection.

Includes per-class and macro metrics aligned with xView2 competition scoring.
"""

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from typing import Dict, List


DAMAGE_CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]


def compute_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    class_names: List[str] = DAMAGE_CLASSES,
) -> Dict[str, float]:
    """
    Compute classification metrics for damage detection.

    Args:
        preds: Predicted class indices (N,)
        targets: Ground truth class indices (N,)
        class_names: List of class name strings

    Returns:
        Dictionary of metric name -> value
    """
    metrics = {}

    # Per-class F1
    f1_per_class = f1_score(targets, preds, average=None,
                            labels=list(range(len(class_names))),
                            zero_division=0)
    for i, name in enumerate(class_names):
        metrics[f"f1_{name}"] = float(f1_per_class[i])

    # Macro / weighted averages
    metrics["f1_macro"] = float(f1_score(targets, preds, average="macro", zero_division=0))
    metrics["f1_weighted"] = float(f1_score(targets, preds, average="weighted", zero_division=0))
    metrics["precision_macro"] = float(precision_score(targets, preds, average="macro", zero_division=0))
    metrics["recall_macro"] = float(recall_score(targets, preds, average="macro", zero_division=0))

    # Accuracy
    metrics["accuracy"] = float(np.mean(preds == targets))

    # xView2 harmonic mean score (F1 of minor+major+destroyed)
    damage_mask = targets > 0
    if damage_mask.sum() > 0:
        metrics["xview2_damage_f1"] = float(
            f1_score(targets[damage_mask], preds[damage_mask],
                     average="macro", zero_division=0)
        )

    return metrics


def print_classification_report(preds: np.ndarray, targets: np.ndarray) -> None:
    """Print a full sklearn classification report."""
    print(classification_report(targets, preds, target_names=DAMAGE_CLASSES, zero_division=0))


def get_confusion_matrix(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Return confusion matrix as numpy array."""
    return confusion_matrix(targets, preds, labels=list(range(len(DAMAGE_CLASSES))))
