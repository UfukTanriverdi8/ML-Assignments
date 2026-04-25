# BBM409 Assignment 3: Anomaly Detection in Machinery Sounds

**Course:** BBM409 - Introduction to Machine Learning Lab, Spring 2026, Hacettepe University
**Deadline:** April 27, 2026

Unsupervised anomaly detection on the [DCASE 2020 Challenge Task 2](https://dcase.community/challenge2020/task2-unsupervised-detection-of-anomalous-sounds) development dataset. Four autoencoder architectures are trained on normal machine sounds; at test time, reconstruction error is the anomaly score. Primary metric: ROC-AUC.

---

## Project structure

```
ML-A3/
├── reportcode.ipynb      # main deliverable: narrative, results, plots
├── pyproject.toml        # uv-managed dependencies
├── CLAUDE.md             # development notes and decisions
├── code/
│   ├── config.py         # paths, constants, default hyperparameters
│   ├── data.py           # file listing, label extraction
│   ├── features.py       # mel / STFT / wavelet extraction + caching
│   ├── models_scratch.py # NumPy MLP autoencoders (backprop from scratch)
│   ├── models_torch.py   # PyTorch CNN autoencoders
│   ├── train.py          # training loops
│   ├── evaluate.py       # AUC, reconstruction error
│   └── viz.py            # plotting helpers
├── cache/                # cached .npz features (gitignored)
├── checkpoints/          # saved model weights (gitignored)
├── data/                 # DCASE dataset (not included)
└── figures/              # saved plots (gitignored)
```

---

## Environment setup

**Requirements:** Python 3.12, [uv](https://github.com/astral-sh/uv), CUDA 12.8-compatible GPU (tested on NVIDIA RTX 5000 Ada).

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create the virtual environment and install dependencies

```bash
uv sync
```

This installs all dependencies from `pyproject.toml`, including PyTorch 2.11 from the CUDA 12.8 index. The venv is created at `.venv/`.

### 3. Activate the environment

```bash
source .venv/bin/activate
```

### 4. Point to the dataset

Open `code/config.py` and verify `DATA_ROOT` points to your copy of the DCASE 2020 Task 2 development dataset. The expected layout inside that directory:

```
data/
├── ToyCar/
│   ├── train/   # normal WAVs only
│   └── test/    # normal + anomalous WAVs
├── pump/
├── fan/
└── ...
```

Filename format: `{normal|anomaly}_id_{MachineID}_{ClipID}.wav`

### 5. Run the notebook

```bash
uv run jupyter lab
```

Open `reportcode.ipynb` and run all cells. Feature extraction on first run takes a few minutes and writes compressed caches to `cache/` so subsequent runs are fast.

> **GPU note:** the training cells pin to GPU 0 via `os.environ["CUDA_VISIBLE_DEVICES"] = "0"`. If you are on a different GPU, change that value before importing torch.

---

## Architectures

| # | Model | Framework | Input |
|---|-------|-----------|-------|
| 1 | Single-layer autoencoder (input -> bottleneck -> output) | NumPy (from scratch) | mel / wavelet |
| 2 | MLP with 2 hidden layers | NumPy (from scratch) | mel / wavelet |
| 3 | Shallow CNN (1 conv + 1 FC) | PyTorch | mel / STFT / wavelet |
| 4 | Deep CNN (2 conv + 2 FC) | PyTorch | mel / STFT / wavelet |

STFT features with NumPy models were skipped (512-bin flat dim makes CPU matrix ops impractical).

## Feature representations

| Feature | Library | Notes |
|---------|---------|-------|
| Mel spectrogram | librosa | 64 mel bins, N_FFT=1024, hop=512 |
| STFT spectrogram | librosa | 512 frequency bins, same window/hop |
| Wavelet scalogram | pywt (CWT) | Morlet wavelet, 64 scales, downsampled by 512 |

## Key results (ToyCar, default hyperparameters)

| Model | mel AUC | stft AUC | wavelet AUC |
|-------|---------|----------|-------------|
| MLP-1layer | 0.5057 | N/A | 0.4489 |
| MLP-2hidden | 0.5856 | N/A | 0.5613 |
| CNN-shallow | 0.5987 | 0.6129 | 0.5813 |
| CNN-deep | **0.6387** | 0.6220 | 0.5616 |

Best tuned result: mel + CNN-deep, batch_size=32, lr_decay=0.99 -> ~0.65 AUC.
