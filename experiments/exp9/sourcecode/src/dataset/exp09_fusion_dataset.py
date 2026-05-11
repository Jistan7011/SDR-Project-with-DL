from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.common import CLASS_NAMES


class Exp09FusionDataset(Dataset):
    def __init__(self, root: str | Path, split: str, preload: bool = False):
        self.root = Path(root) / split
        self.files = sorted(self.root.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {self.root}")
        self.class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
        self.cache = [self.load_sample(path) for path in self.files] if preload else None

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if self.cache is None:
            time, spectral, evidence, label = self.load_sample(self.files[index])
        else:
            time, spectral, evidence, label = self.cache[index]
        return {
            "time": torch.from_numpy(time.copy()),
            "spectral": torch.from_numpy(spectral.copy()),
            "evidence": torch.from_numpy(evidence.copy()),
        }, torch.tensor(label, dtype=torch.long)

    def load_sample(self, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        data = np.load(path, allow_pickle=False)
        time = data["iq"].astype(np.float32)
        spectral = data["spectral_feature"].astype(np.float32)
        evidence = data["evidence_feature"].astype(np.float32)
        label = self.class_to_idx[str(data["modulation"])]
        return time, spectral, evidence, label
