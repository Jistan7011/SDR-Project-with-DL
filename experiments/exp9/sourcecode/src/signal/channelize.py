from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import firwin, lfilter, resample_poly


def channelize_and_downsample(
    iq: np.ndarray,
    sample_rate: float,
    channel_center_hz: float,
    channel_bandwidth_hz: float,
    target_sample_rate: float = 160_000.0,
) -> tuple[np.ndarray, float]:
    """Move a real SDR capture channel to DC, filter it, and resample it."""
    arr = np.asarray(iq, dtype=np.complex64)
    t = np.arange(len(arr), dtype=np.float64) / float(sample_rate)
    shifted = arr * np.exp(-1j * 2.0 * np.pi * float(channel_center_hz) * t)
    cutoff = min(float(channel_bandwidth_hz) / 2.0, float(sample_rate) * 0.45)
    taps = firwin(257, cutoff=cutoff, fs=float(sample_rate))
    filtered = lfilter(taps, [1.0], shifted)
    if abs(float(target_sample_rate) - float(sample_rate)) < 1e-6:
        return filtered.astype(np.complex64), float(sample_rate)
    frac = Fraction(float(target_sample_rate) / float(sample_rate)).limit_denominator(10_000)
    downsampled = resample_poly(filtered, frac.numerator, frac.denominator)
    return downsampled.astype(np.complex64), float(sample_rate) * frac.numerator / frac.denominator


def estimate_snr_db(noise_iq: np.ndarray, active_iq: np.ndarray, eps: float = 1e-12) -> dict[str, float]:
    noise_power = float(np.mean(np.abs(np.asarray(noise_iq, dtype=np.complex64)) ** 2))
    active_power = float(np.mean(np.abs(np.asarray(active_iq, dtype=np.complex64)) ** 2))
    signal_power = max(active_power - noise_power, eps)
    snr_db = 10.0 * float(np.log10(signal_power / max(noise_power, eps)))
    return {
        "noise_power": noise_power,
        "active_power": active_power,
        "signal_power": signal_power,
        "estimated_snr_db": snr_db,
    }


def spectrum_summary(iq: np.ndarray, sample_rate: float, threshold_db: float = 12.0) -> dict[str, float | bool]:
    arr = np.asarray(iq, dtype=np.complex64)
    if len(arr) == 0:
        return {"noise_floor_db": float("nan"), "peak_frequency_hz": float("nan"), "peak_db": float("nan"), "interference_flag": False}
    n = min(262_144, len(arr))
    window = np.hanning(n)
    spectrum = np.fft.fftshift(np.fft.fft(arr[:n] * window))
    power_db = 20.0 * np.log10(np.abs(spectrum) + 1e-12)
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / float(sample_rate)))
    floor = float(np.median(power_db))
    peak_idx = int(np.argmax(power_db))
    peak_db = float(power_db[peak_idx])
    return {
        "noise_floor_db": floor,
        "peak_frequency_hz": float(freqs[peak_idx]),
        "peak_db": peak_db,
        "interference_flag": bool((peak_db - floor) >= float(threshold_db)),
    }
