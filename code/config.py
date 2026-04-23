"""Central configuration: paths, constants, and default hyperparameters."""

from pathlib import Path

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT    = PROJECT_ROOT / "data"
CACHE_DIR    = PROJECT_ROOT / "cache"
CKPT_DIR     = PROJECT_ROOT / "checkpoints"
FIG_DIR      = PROJECT_ROOT / "figures"

for _d in (CACHE_DIR, CKPT_DIR, FIG_DIR):
    _d.mkdir(exist_ok=True)

# ── Audio ────────────────────────────────────────────────────────────────────
SR = 16_000          # native sample rate; always load at 16 kHz
CLIP_DURATION = 10   # seconds

# ── Mel spectrogram ──────────────────────────────────────────────────────────
N_FFT    = 1024
HOP_LEN  = 512
N_MELS   = 64        # frequency bins
FMAX     = SR // 2   # 8 kHz

# ── STFT spectrogram ─────────────────────────────────────────────────────────
# same N_FFT / HOP_LEN, keep only 1..N_FFT//2 bins (drop DC + Nyquist)
STFT_BINS = N_FFT // 2  # 512 bins

# ── Wavelet scalogram ────────────────────────────────────────────────────────
WAVELET    = "morl"   # Morlet (complex Morlet requires 'cmor' + bandwidth params)
N_SCALES   = 64       # number of CWT scales -- matches N_MELS for fair comparison
# CWT produces one coefficient per audio sample, so a 10s clip = 160 000 time points.
# Downsampling by 512 (same hop as mel) gives ~312 frames, matching mel's time axis.
CWT_DOWNSAMPLE = 512

# ── Training defaults ─────────────────────────────────────────────────────────
DEFAULT_LR         = 0.01
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS     = 50
DEFAULT_ACTIVATION = "relu"   # options: sigmoid | tanh | relu
DEFAULT_LOSS       = "mse"    # options: mse | bce

# ── Autoencoder bottleneck ────────────────────────────────────────────────────
BOTTLENECK_DIM = 32
