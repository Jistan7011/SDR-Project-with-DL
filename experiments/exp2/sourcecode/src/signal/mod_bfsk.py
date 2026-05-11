from __future__ import annotations

import numpy as np


def generate_bfsk(
    bits: np.ndarray,
    samples_per_symbol: int,
    sample_rate: float,
    f0: float = 80_000.0,
    f1: float = 120_000.0,
) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.float32).ravel()
    freq = np.repeat(np.where(bits > 0.5, f1, f0), samples_per_symbol)
    phase = np.cumsum(2.0 * np.pi * freq / float(sample_rate))
    return np.exp(1j * phase).astype(np.complex64)
