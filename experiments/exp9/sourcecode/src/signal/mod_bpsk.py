from __future__ import annotations

import numpy as np


def generate_bpsk(bits: np.ndarray, samples_per_symbol: int, sample_rate: float, carrier_freq: float = 100_000.0) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.float32).ravel()
    n = len(bits) * samples_per_symbol
    t = np.arange(n, dtype=np.float32) / float(sample_rate)
    symbols = np.repeat(np.where(bits > 0.5, -1.0, 1.0), samples_per_symbol)
    carrier = np.exp(1j * 2.0 * np.pi * carrier_freq * t)
    return (symbols * carrier).astype(np.complex64)
