from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.dataset.iq_dataset import IQDataset


class BinaryIQDataset(IQDataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        augment: bool = False,
        augmentation: dict[str, object] | None = None,
        feature_mode: str = "iq",
        preload: bool = False,
    ):
        super().__init__(root, split, augment=augment, augmentation=augmentation, feature_mode=feature_mode, preload=False)
        self.cache = None
        if preload:
            self.cache = [self.load_raw_sample(path) for path in self.files]

    def load_raw_sample(self, path: Path) -> tuple[np.ndarray, int]:
        data = np.load(path, allow_pickle=False)
        iq = data["iq"].astype(np.float32)
        label = int(data["label"])
        return iq, label


def binary_sample_metadata(path: str | Path) -> dict[str, object]:
    data = np.load(path, allow_pickle=False)
    return {k: data[k].tolist() if hasattr(data[k], "tolist") else str(data[k]) for k in data.files if k != "iq"}
