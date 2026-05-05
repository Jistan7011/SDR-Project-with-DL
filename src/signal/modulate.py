from __future__ import annotations

import numpy as np

from src.signal.mod_bask import generate_bask
from src.signal.mod_bfsk import generate_bfsk
from src.signal.mod_bpsk import generate_bpsk


def modulate_bits(
    modulation: str,
    bits: np.ndarray,
    samples_per_symbol: int,
    sample_rate: float,
    params: dict[str, float] | None = None,
) -> np.ndarray:
    params = params or {}
    mod = modulation.upper()
    if mod == "BASK":
        return generate_bask(bits, samples_per_symbol, sample_rate, **params)
    if mod == "BFSK":
        return generate_bfsk(bits, samples_per_symbol, sample_rate, **params)
    if mod == "BPSK":
        return generate_bpsk(bits, samples_per_symbol, sample_rate, **params)
    raise ValueError(f"Unsupported modulation: {modulation}")
