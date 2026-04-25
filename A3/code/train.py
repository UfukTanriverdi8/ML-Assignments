"""Training loops for both NumPy and PyTorch autoencoders.

Both trainers share the same interface and return a dict with loss history
and other metadata. This avoids separate training functions per architecture.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from models_scratch import NumpyAutoencoder


# ── NumPy trainer ─────────────────────────────────────────────────────────────

def train_numpy(
    model: NumpyAutoencoder,
    X_train: np.ndarray,
    epochs: int,
    batch_size: int,
    lr_decay: float = 1.0,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train a NumpyAutoencoder.

    Parameters
    ----------
    model : NumpyAutoencoder
    X_train : np.ndarray, shape (N, D) -- already normalised
    epochs : int
    batch_size : int
    seed : int
    verbose : bool

    Returns
    -------
    history : dict with keys 'train_loss' (list of floats, one per epoch)
    """
    rng = np.random.default_rng(seed)
    history: Dict[str, List[float]] = {"train_loss": []}

    iterator = tqdm(range(epochs), desc="NumPy train", disable=not verbose)
    for _ in iterator:
        loss = model.train_epoch(X_train, batch_size=batch_size, rng=rng)
        history["train_loss"].append(loss)
        if verbose:
            iterator.set_postfix(loss=f"{loss:.4f}")
        model.lr *= lr_decay  # no-op when lr_decay=1.0

    return history


# ── PyTorch trainer ───────────────────────────────────────────────────────────

def train_torch(
    model: nn.Module,
    X_train: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    loss_fn: str = "mse",
    lr_decay: float = 1.0,
    device: Optional[torch.device] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train a PyTorch autoencoder.

    Parameters
    ----------
    model : nn.Module -- already moved to device
    X_train : np.ndarray, shape (N, D) -- already normalised
    epochs : int
    batch_size : int
    lr : float
    loss_fn : str -- 'mse' or 'bce'
    device : torch.device
    seed : int
    verbose : bool

    Returns
    -------
    history : dict with keys 'train_loss'
    """
    if device is None:
        device = torch.device("cpu")

    torch.manual_seed(seed)

    criterion = nn.MSELoss() if loss_fn == "mse" else nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=lr_decay)

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    dataset  = TensorDataset(X_tensor)
    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history: Dict[str, List[float]] = {"train_loss": []}
    model.train()

    iterator = tqdm(range(epochs), desc="PyTorch train", disable=not verbose)
    for _ in iterator:
        epoch_loss = 0.0
        n_batches  = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimiser.zero_grad()
            x_hat = model(batch)
            loss  = criterion(x_hat, batch)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()
            n_batches  += 1

        mean_loss = epoch_loss / n_batches
        history["train_loss"].append(mean_loss)
        scheduler.step()  # no-op when lr_decay=1.0 (gamma=1.0 multiplies by 1)
        if verbose:
            iterator.set_postfix(loss=f"{mean_loss:.4f}")

    return history
