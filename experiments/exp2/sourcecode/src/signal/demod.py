from __future__ import annotations

import numpy as np

from src.signal.demod_bask import demod_bask
from src.signal.demod_bfsk import demod_bfsk
from src.signal.demod_bpsk import demod_bpsk
from src.signal.frame import parse_frame


def demodulate(modulation: str, iq: np.ndarray, samples_per_symbol: int, sample_rate: float) -> np.ndarray:
    mod = modulation.upper()
    if mod == "BASK":
        return demod_bask(iq, samples_per_symbol)
    if mod == "BFSK":
        return demod_bfsk(iq, samples_per_symbol)
    if mod == "BPSK":
        return demod_bpsk(iq, samples_per_symbol, sample_rate=sample_rate)
    raise ValueError(f"Unsupported modulation: {modulation}")


def recover_payload(modulation: str, iq: np.ndarray, samples_per_symbol: int, sample_rate: float, payload_bytes: int = 1) -> dict[str, object]:
    bits = demodulate(modulation, iq, samples_per_symbol, sample_rate)
    result = parse_frame(bits, payload_bytes=payload_bytes)
    result["recovered_bits"] = bits
    return result
