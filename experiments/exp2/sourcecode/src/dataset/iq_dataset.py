from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.common import CLASS_NAMES
from src.signal.processing import complex64_from_channels, instantaneous_frequency_channel, iq_to_channels


class IQDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        augment: bool = False,
        augmentation: dict[str, object] | None = None,
        feature_mode: str = "iq",
    ):
        self.root = Path(root) / split
        self.files = sorted(self.root.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {self.root}")
        self.class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
        self.augment = augment
        self.augmentation = augmentation or {}
        self.feature_mode = feature_mode
        self.rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        data = np.load(self.files[index], allow_pickle=False)
        iq = data["iq"].astype(np.float32)
        if self.augment:
            iq = self.apply_augmentation(iq)
        iq = self.apply_feature_mode(iq)
        label = self.class_to_idx[str(data["modulation"])]
        return torch.from_numpy(iq), torch.tensor(label, dtype=torch.long)

    def apply_feature_mode(self, channels: np.ndarray) -> np.ndarray:
        key = self.feature_mode.lower().replace("-", "_")
        if key in {"iq", "i_q"}:
            return channels.astype(np.float32)
        complex_iq = complex64_from_channels(channels)
        if key in {"iq_ifreq", "iq_instfreq", "iq_instantaneous_frequency"}:
            inst_freq = instantaneous_frequency_channel(complex_iq)
            return np.concatenate([channels[:2], inst_freq[None, :]], axis=0).astype(np.float32)
        if key in {"iq_mag", "iq_magnitude"}:
            magnitude = np.abs(complex_iq).astype(np.float32)
            magnitude = (magnitude - np.mean(magnitude)) / (np.std(magnitude) + 1e-8)
            return np.concatenate([channels[:2], magnitude[None, :]], axis=0).astype(np.float32)
        raise ValueError(f"Unsupported feature_mode: {self.feature_mode}")

    def apply_augmentation(self, channels: np.ndarray) -> np.ndarray:
        iq = complex64_from_channels(channels)
        window_size = channels.shape[1]
        if bool(self.augmentation.get("time_shift", True)):
            max_shift = int(self.augmentation.get("max_time_shift", 64))
            if max_shift > 0:
                shift = int(self.rng.integers(-max_shift, max_shift + 1))
                iq = np.roll(iq, shift)
        if bool(self.augmentation.get("phase_rotation", True)):
            phase = float(self.rng.uniform(-np.pi, np.pi))
            iq = iq * np.exp(1j * phase)
        if bool(self.augmentation.get("frequency_jitter", True)):
            max_cycles = float(self.augmentation.get("max_frequency_jitter_cycles", 3.0))
            cycles = float(self.rng.uniform(-max_cycles, max_cycles))
            n = np.arange(window_size, dtype=np.float32)
            iq = iq * np.exp(1j * 2.0 * np.pi * cycles * n / max(window_size, 1))
        augmented = iq_to_channels(iq.astype(np.complex64), window_size)
        if bool(self.augmentation.get("amplitude_scale", True)):
            min_gain = float(self.augmentation.get("min_amplitude_scale", 0.75))
            max_gain = float(self.augmentation.get("max_amplitude_scale", 1.25))
            gain = float(self.rng.uniform(min_gain, max_gain))
            augmented = augmented * gain
        return augmented.astype(np.float32)


def sample_metadata(path: str | Path) -> dict[str, object]:
    data = np.load(path, allow_pickle=False)
    return {k: data[k].tolist() if hasattr(data[k], "tolist") else str(data[k]) for k in data.files if k != "iq"}
