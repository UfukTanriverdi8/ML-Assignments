"""Save and load trained models to/from checkpoints/."""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from config import CKPT_DIR


def _safe(feat: str, model_name: str) -> str:
    return f"{feat}__{model_name}".replace("-", "_").replace(" ", "_")


# ── NumPy ─────────────────────────────────────────────────────────────────────

def save_numpy(model: Any, feat: str, model_name: str) -> Path:
    path = CKPT_DIR / f"{_safe(feat, model_name)}.npz"
    arrays: Dict[str, Any] = {}
    for i, (W, b) in enumerate(zip(model.Ws, model.bs)):
        arrays[f"W{i}"] = W
        arrays[f"b{i}"] = b
    arrays["cfg_input_dim"]     = np.array(model.input_dim)
    arrays["cfg_hidden_dims"]   = np.array(model.hidden_dims)
    arrays["cfg_bottleneck_dim"] = np.array(model.bottleneck_dim)
    arrays["cfg_lr"]            = np.array(model.lr)
    arrays["cfg_grad_clip"]     = np.array(model.grad_clip)
    arrays["cfg_activation"]    = np.array(model.activation)
    np.savez(path, **arrays)
    return path


def load_numpy(feat: str, model_name: str) -> Any:
    from models_scratch import NumpyAutoencoder
    path = CKPT_DIR / f"{_safe(feat, model_name)}.npz"
    d = np.load(path, allow_pickle=True)
    model = NumpyAutoencoder(
        input_dim=int(d["cfg_input_dim"]),
        hidden_dims=d["cfg_hidden_dims"].tolist(),
        bottleneck_dim=int(d["cfg_bottleneck_dim"]),
        activation=str(d["cfg_activation"]),
        lr=float(d["cfg_lr"]),
        grad_clip=float(d["cfg_grad_clip"]),
    )
    for i in range(model.n_layers):
        model.Ws[i] = d[f"W{i}"]
        model.bs[i] = d[f"b{i}"]
    return model


# ── PyTorch ───────────────────────────────────────────────────────────────────

def save_torch(model: nn.Module, feat: str, model_name: str) -> Path:
    path = CKPT_DIR / f"{_safe(feat, model_name)}.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "input_shape":    model.input_shape,
        "conv_channels":  model.conv_channels,
        "fc_dims":        model.fc_dims,
        "bottleneck_dim": model.bottleneck_dim,
        "kernel_size":    model.kernel_size,
        "stride":         model.stride,
    }, path)
    return path


def load_torch(feat: str, model_name: str, device: torch.device) -> nn.Module:
    from models_torch import ConvAutoencoder
    path = CKPT_DIR / f"{_safe(feat, model_name)}.pt"
    data = torch.load(path, map_location=device)
    model = ConvAutoencoder(
        input_shape=data["input_shape"],
        conv_channels=data["conv_channels"],
        fc_dims=data["fc_dims"],
        bottleneck_dim=data["bottleneck_dim"],
        kernel_size=data["kernel_size"],
        stride=data["stride"],
    ).to(device)
    model.load_state_dict(data["state_dict"])
    return model


# ── Save all ──────────────────────────────────────────────────────────────────

def save_all(results: dict) -> None:
    """Save every model in the results dict."""
    for (feat, model_name), entry in results.items():
        model = entry["model"]
        if hasattr(model, "Ws"):
            path = save_numpy(model, feat, model_name)
        else:
            path = save_torch(model, feat, model_name)
        print(f"  saved {path.name}")
