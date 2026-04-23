"""Feature extraction (mel, STFT, wavelet) with disk caching.

Each feature type is extracted once per machine type + split and saved to
cache/ as an .npz file. Subsequent calls load from cache.

Output arrays are 2-D: shape (N_clips, flat_feature_dim).
The flattened representation is used by both NumPy MLPs and PyTorch CNNs
(CNNs reshape internally).
"""

from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np
import pywt
from tqdm.auto import tqdm

from config import (
    CACHE_DIR,
    SR,
    N_FFT,
    HOP_LEN,
    N_MELS,
    FMAX,
    STFT_BINS,
    WAVELET,
    N_SCALES,
    CWT_DOWNSAMPLE,
)


# ── Low-level per-clip extractors ─────────────────────────────────────────────

def _mel_spec(y: np.ndarray) -> np.ndarray:
    """Mel spectrogram in dB, shape (N_MELS, T)."""
    S = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LEN, n_mels=N_MELS, fmax=FMAX
    )
    return librosa.power_to_db(S, ref=np.max)


def _stft_spec(y: np.ndarray) -> np.ndarray:
    """STFT magnitude in dB, shape (STFT_BINS, T).

    Only the positive frequencies are kept (bins 1..N_FFT//2), discarding
    DC and the Nyquist bin which carry no audio information.
    """
    D = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LEN))
    D = D[1 : STFT_BINS + 1, :]   # shape: (512, T)
    return librosa.amplitude_to_db(D, ref=np.max)


def _wavelet_scalogram(y: np.ndarray) -> np.ndarray:
    """CWT scalogram (power), shape (N_SCALES, T_down).

    pywt.cwt returns coefficients for each scale. We take the absolute value
    (amplitude), square to get power, downsample in time by CWT_DOWNSAMPLE to
    keep the array manageable, then convert to dB.
    """
    # Logarithmically spaced scales: large scale = low frequency, small = high.
    scales = np.geomspace(1, 128, num=N_SCALES)
    coeffs, _ = pywt.cwt(y, scales, WAVELET, sampling_period=1.0 / SR)
    power = np.abs(coeffs) ** 2                     # (N_SCALES, T)
    power = power[:, ::CWT_DOWNSAMPLE]              # downsample time
    power_db = 10 * np.log10(power + 1e-10)
    return power_db                                  # (N_SCALES, T_down)


# ── Batch extraction with caching ─────────────────────────────────────────────

def _load_audio(path: Path) -> np.ndarray:
    """Load a WAV at native SR (16 kHz), return mono float32 array."""
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    return y


def _extract_all(
    file_list: List[Path],
    feature_type: str,
) -> np.ndarray:
    """Extract features for every clip in file_list and stack into (N, D).

    All clips are trimmed / zero-padded to the same time length before
    flattening, so every row has identical dimensionality.
    """
    extractor = {
        "mel": _mel_spec,
        "stft": _stft_spec,
        "wavelet": _wavelet_scalogram,
    }[feature_type]

    frames: List[np.ndarray] = []
    for path in tqdm(file_list, desc=f"  {feature_type}", leave=False):
        y    = _load_audio(path)
        spec = extractor(y)      # (F, T)
        frames.append(spec)

    # Align all clips to the same time length (use the minimum across the batch)
    T_min = min(f.shape[1] for f in frames)
    aligned = [f[:, :T_min] for f in frames]

    # Flatten: (F, T) -> (F*T,)
    return np.stack([a.ravel() for a in aligned], axis=0)  # (N, F*T)


def _cache_path(machine_type: str, split: str, feature_type: str) -> Path:
    return CACHE_DIR / f"{machine_type}_{split}_{feature_type}.npz"


def load_features(
    machine_type: str,
    split: str,
    feature_type: str,
    file_list: List[Path],
    force_recompute: bool = False,
) -> np.ndarray:
    """Load features from cache or compute and cache them.

    Parameters
    ----------
    machine_type : str
    split : str
        'train' or 'test'.
    feature_type : str
        'mel', 'stft', or 'wavelet'.
    file_list : list of Path
        Ordered list of audio files for this split.
    force_recompute : bool
        If True, ignore cache and recompute.

    Returns
    -------
    X : np.ndarray, shape (N, D)
        Raw (un-normalized) features. Normalize before feeding to a model.
    """
    cache = _cache_path(machine_type, split, feature_type)

    if cache.exists() and not force_recompute:
        return np.load(cache)["X"]

    print(f"Extracting {feature_type} features for {machine_type}/{split} ...")
    X = _extract_all(file_list, feature_type)
    np.savez_compressed(cache, X=X)
    print(f"  -> shape {X.shape}, saved to {cache.name}")
    return X


# ── Normalisation ─────────────────────────────────────────────────────────────

def fit_scaler(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean and std from the training set.

    Parameters
    ----------
    X_train : np.ndarray, shape (N, D)

    Returns
    -------
    mean, std : each shape (D,)
        std is clipped to 1e-8 to avoid division by zero for silent features.
    """
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0).clip(1e-8)
    return mean, std


def normalize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Standardize X using pre-computed mean and std (zero mean, unit variance)."""
    return (X - mean) / std
