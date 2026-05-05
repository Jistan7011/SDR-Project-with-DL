from __future__ import annotations

import numpy as np


def generate_bpsk(
    bits: np.ndarray,
    samples_per_symbol: int,
    sample_rate: float,
    carrier_freq: float = 100_000.0,
    phase0: float = 0.0,
    phase1: float = np.pi,
) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.float32).ravel()
    n = len(bits) * samples_per_symbol
    t = np.arange(n, dtype=np.float32) / float(sample_rate)
    symbol_phase = np.repeat(np.where(bits > 0.5, phase1, phase0), samples_per_symbol)
    carrier = np.exp(1j * 2.0 * np.pi * carrier_freq * t)
    return (np.exp(1j * symbol_phase) * carrier).astype(np.complex64)
