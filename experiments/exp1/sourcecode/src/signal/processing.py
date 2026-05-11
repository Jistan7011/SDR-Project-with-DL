from __future__ import annotations

import numpy as np


def normalize_iq(iq: np.ndarray) -> np.ndarray:
    arr = np.asarray(iq, dtype=np.complex64)
    arr = arr - np.mean(arr)
    scale = np.std(arr)
    return (arr / (scale + 1e-8)).astype(np.complex64)


def iq_to_channels(iq: np.ndarray, window_size: int) -> np.ndarray:
    arr = normalize_iq(iq)
    if len(arr) < window_size:
        arr = np.pad(arr, (0, window_size - len(arr)))
    arr = arr[:window_size]
    return np.stack([arr.real, arr.imag], axis=0).astype(np.float32)


def complex64_from_channels(channels: np.ndarray) -> np.ndarray:
    x = np.asarray(channels)
    if x.shape[0] != 2:
        raise ValueError(f"Expected channels shape [2, N], got {x.shape}")
    return (x[0] + 1j * x[1]).astype(np.complex64)
