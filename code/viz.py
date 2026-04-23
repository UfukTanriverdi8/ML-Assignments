"""Plotting helpers for the report notebook."""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curve(
    history: Dict,
    title: str = "Training loss",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot training loss over epochs."""
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(history["train_loss"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def plot_roc(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    label: str = "",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot a single ROC curve."""
    if ax is None:
        _, ax = plt.subplots()
    lbl = f"{label} (AUC={auc:.3f})" if label else f"AUC={auc:.3f}"
    ax.plot(fpr, tpr, label=lbl)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_score_distribution(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: Optional[float] = None,
    title: str = "Anomaly score distribution",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Histogram of anomaly scores coloured by ground truth label."""
    if ax is None:
        _, ax = plt.subplots()
    ax.hist(scores[labels == 0], bins=50, alpha=0.6, label="Normal", density=True)
    ax.hist(scores[labels == 1], bins=50, alpha=0.6, label="Anomaly", density=True)
    if threshold is not None:
        ax.axvline(threshold, color="red", linestyle="--", label=f"Threshold={threshold:.4f}")
    ax.set_xlabel("Reconstruction MSE")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_spectrogram(
    spec: np.ndarray,
    title: str = "Spectrogram",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Display a 2-D spectrogram (F x T) as an image."""
    if ax is None:
        _, ax = plt.subplots()
    img = ax.imshow(spec, aspect="auto", origin="lower", cmap="magma")
    plt.colorbar(img, ax=ax, label="dB")
    ax.set_xlabel("Time frame")
    ax.set_ylabel("Frequency bin")
    ax.set_title(title)
    return ax


def plot_weights(
    W: np.ndarray,
    input_shape: Tuple[int, int],
    n_units: int = 16,
    title: str = "First-layer weights",
) -> plt.Figure:
    """Reshape and plot the first n_units columns of W as spectrograms.

    Parameters
    ----------
    W : np.ndarray, shape (input_dim, hidden_dim)
        First-layer weight matrix of a NumPy MLP.
    input_shape : (freq_bins, time_frames)
    n_units : int
        How many weight vectors to visualise.
    """
    n_cols = 8
    n_rows = (n_units + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = np.array(axes).ravel()

    for i in range(n_units):
        w = W[:, i].reshape(input_shape)
        axes[i].imshow(w, aspect="auto", origin="lower", cmap="RdBu_r")
        axes[i].axis("off")
    for j in range(n_units, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title)
    fig.tight_layout()
    return fig
