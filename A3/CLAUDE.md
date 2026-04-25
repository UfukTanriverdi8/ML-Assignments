# BBM409 Assignment 3: Anomaly Detection in Machinery Sounds

## Project overview

Course assignment for BBM409 (Introduction to Machine Learning Lab, Spring 2026, Hacettepe University). Deadline: April 27, 2026.

The task is **unsupervised anomaly detection** on the DCASE 2020 Challenge Task 2 development dataset. Training data contains only "normal" machine sounds; test data contains both normal and anomalous sounds. The model must decide, for each test clip, whether it is normal or anomalous.

**Approach: autoencoders.** Each model is trained to reconstruct its input, minimizing reconstruction error on normal data. At test time, reconstruction error serves as the anomaly score: normal sounds reconstruct well, anomalies reconstruct poorly. ROC-AUC is the primary metric (no need to pick a threshold up front).

## Scope

- **Primary development machine type: ToyCar** (4000 train clips, balanced test: 1400 normal + 1059 anomalous). Chosen because the balanced test set gives cleaner AUC estimates and the ample training data helps the autoencoders learn.
- **Possible extension**: run best-configured models on `pump` and/or `fan` at the end to check whether the architecture ranking generalizes to real industrial machines. Only do this if time permits; not required.
- **Do not** train per-machine-ID. Pool all IDs of a machine type into a single autoencoder. Simpler and defensible for a course assignment.

## Four obligatory architectures

All four are autoencoders (input shape == output shape). Varying the depth and layer types:

1. **Single-layer NN** (shallowest autoencoder): input → bottleneck → output. Implement from scratch in NumPy.
2. **MLP with 2 hidden layers**: input → hidden → bottleneck → hidden → output. Implement from scratch in NumPy.
3. **CNN with 1 conv + 1 FC**: shallow convolutional autoencoder. PyTorch.
4. **CNN with 2 conv + 2 FC**: deeper convolutional autoencoder. PyTorch.

The assignment PDF explicitly allows PyTorch for CNNs (including transfer learning). MLPs must be from scratch; the PDF repeatedly says "implement a back-propagation algorithm."

## Three feature representations to compare

- **Mel spectrogram** (librosa). DCASE baseline default. Good starting point.
- **STFT spectrogram** (librosa). Linear frequency axis.
- **Wavelet scalogram** (pywavelets, CWT). Librosa does not do CWT, use pywt.

Extract features once, cache to `cache/` as `.npz`. Do not re-extract on every run.

## Hyperparameters to sweep

Per the PDF (required):
- Learning rate: 0.005 to 0.02
- Batch size: 16 to 128
- Activation: sigmoid, tanh, ReLU
- Loss: MSE (natural for reconstruction), possibly also BCE if inputs are normalized to [0, 1]

Report should include tables of accuracy / AUC vs these parameters. Discuss trends.

**What was actually tested** (CNN-only, GPU; NumPy CPU infeasible for a full sweep):

| Parameter | Values tested | Best |
|---|---|---|
| lr | 0.001, 0.005, 0.01 | 0.005 or 0.01 (plateau) |
| batch_size | 32, 64, 128 | 32 |
| activation | sigmoid, tanh, relu | relu (mel/stft); sigmoid (wavelet) |
| loss_fn | mse | mse only -- BCE excluded (see below) |
| lr_decay | 0.95, 0.99, 1.0 | 0.99 |

**Known gotchas from sweep:**
- `lr=0.02` causes loss divergence on CNNs. Do not use.
- `batch_size=16` impractical with STFT features (GPU stalls, very slow).
- `BCEWithLogitsLoss` requires targets in [0,1]. Standardized features span ~-10 to +63. BCE produces NaN or negative loss. Always use MSE for continuous spectrogram reconstruction.
- `lr_decay=0.95` is too aggressive: effective LR at epoch 50 is `0.01 × 0.95^50 ≈ 0.0009`, landing in underfitting territory.

## Project structure

```
bbm409_a3/
├── reportcode.ipynb     # main deliverable: narrative + results + plots
├── pyproject.toml       # uv-managed
├── CLAUDE.md            # this file
├── code/
│   ├── __init__.py
│   ├── config.py        # paths, constants, defaults
│   ├── data.py          # file listing, label extraction
│   ├── features.py      # mel/stft/wavelet extraction + caching
│   ├── models_scratch.py # numpy MLPs
│   ├── models_torch.py  # pytorch CNN autoencoders
│   ├── train.py         # training loops
│   ├── evaluate.py      # AUC, reconstruction error, thresholds
│   └── viz.py           # plotting helpers
├── cache/               # cached features (gitignored)
├── checkpoints/         # saved model weights (gitignored)
└── figures/             # report figures (gitignored)
```

## Notebook vs .py files

- **`.py` files**: all implementation. Classes, algorithms, numerical code.
- **Notebook**: narrative, markdown explanations, parameter choices, results, plots, tables, discussion. Imports from `code/`.

Keep the notebook readable. No 80-line inline cells doing low-level feature extraction. If it's more than ~20 lines and reusable, it goes in a `.py` file.

## Code conventions

- Python 3.12
- Numpy-style docstrings
- Type hints where they help, not aggressively everywhere
- Prefer clarity over cleverness; this is a homework, not production
- Comments must explain the math where it is non-obvious (backprop derivations, why a specific normalization, etc). The PDF explicitly says: "Comment your code with corresponding mathematical functions."
- Reproducibility: one centrally-seeded RNG. See `SEED = 810` in `config.py`.

## Environment

- Python 3.12, `uv`-managed venv at `.venv/`
- PyTorch 2.11 with CUDA 12.8 (`torch==2.11.0+cu128`)
- Running on anzu lab server (shared), using GPU 0 (NVIDIA RTX 5000 Ada, 32GB)
- Pin to GPU 0 via `os.environ["CUDA_VISIBLE_DEVICES"] = "0"` BEFORE importing torch

## Dataset

- Located at `DATA_ROOT` in `config.py` (set this to the actual path)
- 6 machine types: ToyCar, ToyConveyor, fan, pump, slider, valve
- Each has `train/` (normal only) and `test/` (normal + anomalous, labeled by filename prefix)
- Files: 16 kHz mono WAV, ~10 seconds each
- Filename format: `{normal|anomaly}_id_{MachineID}_{ClipID}.wav`

## Implementation priorities

**DONE** as of April 2026:
1. Full pipeline on ToyCar: all 4 architectures x 3 features (STFT+MLP skipped, infeasible on CPU).
2. Checkpoints saved for all 10 trained models (`checkpoints/`).
3. Hyperparameter sweep completed on 3 best CNN configs.
4. Notebook complete: baseline results, sweep tables + discussion, loss curves, ROC curves, score distribution, weight visualisation, conclusion.

**If continuing:** the only remaining optional item is running best-configured models on `pump`/`fan` to check generalization.

## Baseline AUC results (ToyCar, default hyperparameters)

| Model | mel | stft | wavelet |
|---|---|---|---|
| MLP-1layer | 0.5057 | N/A | 0.4489 |
| MLP-2hidden | 0.5856 | N/A | 0.5613 |
| CNN-shallow | 0.5987 | 0.6129 | 0.5813 |
| CNN-deep | **0.6387** | 0.6220 | 0.5616 |

STFT+NumPy skipped: flat dim 176,128 makes CPU matrix ops take >2 hours per run.

Best tuned result: mel + CNN-deep with batch_size=32, lr_decay=0.99 reaches ~AUC 0.65.

## Reporting requirements (from the assignment PDF)

- Tables of results for each (architecture, feature, hyperparameter setting) combination
- Loss curves (training) and AUC curves for each architecture
- Visualization of learned first-layer weights (for the MLP variants, reshape back to spectrogram shape)
- Comparison of architectures on accuracy, parameter count, and training/test error
- Discussion of parameter effects (learning rate too high/low, batch size, activation choice, etc.)

## Things that were avoided during development (edge cases from PDF, some computational gotchas, and personal preferences)

- Do not write a separate training function per architecture. The PDF says: "do not write separate code for each architecture. If you use n layers, your method should create an n-layer network." Implementations must be parameterized by layer count.
- Do not use `librosa.load` with the default `sr=22050`, pass `sr=None` to keep the native 16 kHz, or `sr=16000` explicitly.
- Do not forget to normalize features before training. Spectrograms in dB scale typically span ~-80 to 0; standardize (zero mean, unit variance) per-feature before feeding to the network.
- Do not commit `cache/`, `checkpoints/`, or `figures/` unless there's a reason. Gitignored.
- Do not sample anomalous clips into training. Train set must be normal-only. The whole point of the task.
- Did not use BCE loss with standardized features -- targets outside [0,1] produce NaN. MSE only.
- Did not use lr=0.02 with CNNs -- causes divergence.
- Training loss history was NOT saved in checkpoints (only weights). If you need loss curves after a kernel restart, re-train with fewer epochs (20 is enough for the curve shape).

## User preferences

Ufuk prefers:
- Concise, to-the-point communication
- No em dashes (they read as AI-generated)
- Understanding what the code does, not just having it written. When adding significant new logic, explain the why in the commit message or cell markdown.
- `uv` for Python env management (already set up)