from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.common import CLASS_NAMES


class IQDataset(Dataset):
    def __init__(self, root: str | Path, split: str):
        self.root = Path(root) / split
        self.files = sorted(self.root.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {self.root}")
        self.class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        data = np.load(self.files[index], allow_pickle=False)
        iq = data["iq"].astype(np.float32)
        label = self.class_to_idx[str(data["modulation"])]
        return torch.from_numpy(iq), torch.tensor(label, dtype=torch.long)


def sample_metadata(path: str | Path) -> dict[str, object]:
    data = np.load(path, allow_pickle=False)
    return {k: data[k].tolist() if hasattr(data[k], "tolist") else str(data[k]) for k in data.files if k != "iq"}
