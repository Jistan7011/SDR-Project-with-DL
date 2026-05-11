from __future__ import annotations

import numpy as np


def add_awgn(iq: np.ndarray, snr_db: float, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    iq = np.asarray(iq, dtype=np.complex64)
    signal_power = float(np.mean(np.abs(iq) ** 2))
    if signal_power <= 0.0:
        return iq.copy()
    snr_linear = 10.0 ** (float(snr_db) / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2.0) * (rng.standard_normal(iq.shape) + 1j * rng.standard_normal(iq.shape))
    return (iq + noise).astype(np.complex64)


def measure_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    noise = np.asarray(noisy) - np.asarray(clean)
    p_signal = float(np.mean(np.abs(clean) ** 2))
    p_noise = float(np.mean(np.abs(noise) ** 2))
    return 10.0 * np.log10(p_signal / (p_noise + 1e-12))
