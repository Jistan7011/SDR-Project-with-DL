from __future__ import annotations

import numpy as np


def demod_bfsk(iq: np.ndarray, samples_per_symbol: int) -> np.ndarray:
    phase = np.unwrap(np.angle(np.asarray(iq)))
    inst = np.diff(phase, prepend=phase[0])
    n_symbols = len(inst) // samples_per_symbol
    values = inst[: n_symbols * samples_per_symbol].reshape(n_symbols, samples_per_symbol).mean(axis=1)
    threshold = (float(values.max()) + float(values.min())) / 2.0 if len(values) else 0.0
    return (values > threshold).astype(np.uint8)
