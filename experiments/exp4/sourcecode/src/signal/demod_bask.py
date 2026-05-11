from __future__ import annotations

import numpy as np


def demod_bask(iq: np.ndarray, samples_per_symbol: int) -> np.ndarray:
    envelope = np.abs(np.asarray(iq))
    n_symbols = len(envelope) // samples_per_symbol
    values = envelope[: n_symbols * samples_per_symbol].reshape(n_symbols, samples_per_symbol).mean(axis=1)
    threshold = (float(values.max()) + float(values.min())) / 2.0 if len(values) else 0.0
    return (values > threshold).astype(np.uint8)
