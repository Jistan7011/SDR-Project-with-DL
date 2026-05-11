from __future__ import annotations

import numpy as np


def generate_bask(bits: np.ndarray, samples_per_symbol: int, sample_rate: float, carrier_freq: float = 100_000.0) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.float32).ravel()
    n = len(bits) * samples_per_symbol
    t = np.arange(n, dtype=np.float32) / float(sample_rate)
    envelope = np.repeat(bits, samples_per_symbol)
    carrier = np.exp(1j * 2.0 * np.pi * carrier_freq * t)
    return (envelope * carrier).astype(np.complex64)
