from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.signal.processing import (
    differential_phase_channel,
    instantaneous_frequency_channel,
    magnitude_channel,
    normalize_iq,
    psd_feature,
)


@dataclass(frozen=True)
class CanonicalizationResult:
    iq: np.ndarray
    time_features: np.ndarray
    spectral_feature: np.ndarray
    metadata: dict[str, float]


def canonicalize_iq(
    channels: np.ndarray,
    sample_rate: float,
    spectral_bins: int = 128,
    symbol_samples: int = 32,
    rms_floor: float = 1e-6,
) -> CanonicalizationResult:
    raw = np.asarray(channels, dtype=np.float32)
    if raw.shape[0] < 2:
        raise ValueError(f"Expected at least I/Q channels, got {raw.shape}")
    iq = (raw[0] + 1j * raw[1]).astype(np.complex64)
    dc = np.mean(iq)
    dc_removed = iq - dc
    rms_before = float(np.sqrt(np.mean(np.abs(dc_removed) ** 2)))
    normalized = dc_removed / max(rms_before, rms_floor)
    slope_before = estimate_phase_slope(normalized)
    n = np.arange(len(normalized), dtype=np.float32)
    corrected = normalized * np.exp(-1j * slope_before * n)
    slope_after = estimate_phase_slope(corrected)
    corrected = normalize_iq(corrected).astype(np.complex64)

    mag = magnitude_channel(corrected)
    ifreq = instantaneous_frequency_channel(corrected)
    dphase = differential_phase_channel(corrected, symbol_samples=symbol_samples)
    time_features = np.stack([corrected.real, corrected.imag, mag, ifreq, dphase], axis=0).astype(np.float32)
    spectral = psd_feature(corrected, bins=spectral_bins).astype(np.float32)
    metadata = {
        "dc_i": float(np.real(dc)),
        "dc_q": float(np.imag(dc)),
        "dc_magnitude": float(np.abs(dc)),
        "rms_before": rms_before,
        "estimated_cfo_hz": float(slope_before * float(sample_rate) / (2.0 * np.pi)),
        "phase_slope_before": float(slope_before),
        "phase_slope_after": float(slope_after),
        "phase_slope_abs_reduction": float(abs(slope_before) - abs(slope_after)),
    }
    return CanonicalizationResult(
        iq=corrected.astype(np.complex64),
        time_features=np.nan_to_num(time_features, nan=0.0, posinf=0.0, neginf=0.0),
        spectral_feature=np.nan_to_num(spectral, nan=0.0, posinf=0.0, neginf=0.0),
        metadata=metadata,
    )


def estimate_phase_slope(iq: np.ndarray) -> float:
    arr = np.asarray(iq, dtype=np.complex64)
    if len(arr) < 2:
        return 0.0
    step = np.angle(arr[1:] * np.conj(arr[:-1])).astype(np.float32)
    finite = step[np.isfinite(step)]
    if len(finite) == 0:
        return 0.0
    return float(np.median(finite))
