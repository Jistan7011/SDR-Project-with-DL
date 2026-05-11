from __future__ import annotations

import numpy as np


def demod_bpsk(iq: np.ndarray, samples_per_symbol: int, carrier_freq: float = 100_000.0, sample_rate: float = 2_400_000.0) -> np.ndarray:
    arr = np.asarray(iq, dtype=np.complex64)
    t = np.arange(len(arr), dtype=np.float32) / float(sample_rate)
    baseband = arr * np.exp(-1j * 2.0 * np.pi * carrier_freq * t)
    n_symbols = len(baseband) // samples_per_symbol
    values = baseband[: n_symbols * samples_per_symbol].reshape(n_symbols, samples_per_symbol).real.mean(axis=1)
    return (values < 0.0).astype(np.uint8)
