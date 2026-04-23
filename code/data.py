"""File listing and label extraction for the DCASE 2020 Task 2 dataset."""

from pathlib import Path
from typing import List, Tuple

import numpy as np

from config import DATA_ROOT


def get_file_list(machine_type: str, split: str) -> List[Path]:
    """Return sorted list of WAV paths for a machine type and split.

    Parameters
    ----------
    machine_type : str
        One of ToyCar, ToyConveyor, fan, pump, slider, valve.
    split : str
        'train' or 'test'.
    """
    directory = DATA_ROOT / machine_type / split
    return sorted(directory.glob("*.wav"))


def get_labels(file_list: List[Path]) -> np.ndarray:
    """Extract binary labels from filenames.

    Returns
    -------
    labels : np.ndarray of int, shape (N,)
        0 = normal, 1 = anomaly.
        Filename must start with 'normal_' or 'anomaly_'.
    """
    labels = []
    for p in file_list:
        name = p.name
        if name.startswith("normal_"):
            labels.append(0)
        elif name.startswith("anomaly_"):
            labels.append(1)
        else:
            raise ValueError(f"Cannot determine label from filename: {name}")
    return np.array(labels, dtype=np.int32)


def train_test_split_files(
    machine_type: str,
) -> Tuple[List[Path], List[Path], np.ndarray]:
    """Return (train_files, test_files, test_labels) for a machine type.

    Train files are all normal. Test files are mixed; labels are derived
    from filenames.
    """
    train_files = get_file_list(machine_type, "train")
    test_files  = get_file_list(machine_type, "test")
    test_labels = get_labels(test_files)
    return train_files, test_files, test_labels
