"""Evaluation utilities: reconstruction error -> anomaly scores -> AUC."""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve

from models_scratch import NumpyAutoencoder


def score_numpy(model: NumpyAutoencoder, X: np.ndarray) -> np.ndarray:
    """Per-sample reconstruction MSE for a NumPy model, shape (N,)."""
    return model.reconstruction_error(X)


def score_torch(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Per-sample reconstruction MSE for a PyTorch model, shape (N,).

    Runs in batches to avoid OOM on large test sets.
    """
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    scores: list = []

    with torch.no_grad():
        for start in range(0, len(X_tensor), batch_size):
            batch = X_tensor[start : start + batch_size].to(device)
            x_hat = model(batch)
            per_sample = ((batch - x_hat) ** 2).mean(dim=1)
            scores.append(per_sample.cpu().numpy())

    return np.concatenate(scores)


def compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute ROC-AUC.

    Parameters
    ----------
    scores : np.ndarray, shape (N,)
        Anomaly scores (higher = more anomalous).
    labels : np.ndarray of int, shape (N,)
        0 = normal, 1 = anomaly.

    Returns
    -------
    auc : float in [0, 1]
    """
    return float(roc_auc_score(labels, scores))


def compute_roc(
    scores: np.ndarray, labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (fpr, tpr, thresholds) for plotting the ROC curve."""
    return roc_curve(labels, scores)


def best_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, float]:
    """Find the threshold that maximises balanced accuracy (Youden's J).

    Returns
    -------
    threshold : float
    balanced_accuracy : float
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j_stat = tpr - fpr
    best_idx = np.argmax(j_stat)
    thr = float(thresholds[best_idx])

    preds = (scores >= thr).astype(int)
    tp = np.sum((preds == 1) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))

    sensitivity = tp / (tp + fn + 1e-10)
    specificity = tn / (tn + fp + 1e-10)
    bal_acc = (sensitivity + specificity) / 2.0

    return thr, float(bal_acc)
