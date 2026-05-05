from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ChannelImpairmentParams:
    enabled: bool = False
    carrier_freq_offset_hz: tuple[float, float] = (-5_000.0, 5_000.0)
    carrier_phase_offset_rad: tuple[float, float] = (0.0, 2.0 * np.pi)
    gain_db: tuple[float, float] = (-6.0, 6.0)
    dc_offset: tuple[float, float] = (-0.02, 0.02)
    time_offset_samples: tuple[int, int] = (0, 0)
    sample_rate_offset_ppm: tuple[float, float] = (-20.0, 20.0)
    iq_gain_imbalance_db: tuple[float, float] = (-1.0, 1.0)
    iq_phase_imbalance_deg: tuple[float, float] = (-3.0, 3.0)
    multipath_enabled: bool = False
    multipath_max_delay_samples: int = 8
    multipath_taps: tuple[int, int] = (2, 4)
    multipath_decay: tuple[float, float] = (0.15, 0.55)


def params_from_config(config: dict[str, Any]) -> ChannelImpairmentParams:
    raw = config.get("channel_impairment", {}) or {}
    return ChannelImpairmentParams(
        enabled=bool(raw.get("enabled", False)),
        carrier_freq_offset_hz=_float_pair(raw.get("carrier_freq_offset_hz"), (-5_000.0, 5_000.0)),
        carrier_phase_offset_rad=_float_pair(raw.get("carrier_phase_offset_rad"), (0.0, 2.0 * np.pi)),
        gain_db=_float_pair(raw.get("gain_db"), (-6.0, 6.0)),
        dc_offset=_float_pair(raw.get("dc_offset"), (-0.02, 0.02)),
        time_offset_samples=_int_pair(raw.get("time_offset_samples"), (0, 0)),
        sample_rate_offset_ppm=_float_pair(raw.get("sample_rate_offset_ppm"), (-20.0, 20.0)),
        iq_gain_imbalance_db=_float_pair(raw.get("iq_gain_imbalance_db"), (-1.0, 1.0)),
        iq_phase_imbalance_deg=_float_pair(raw.get("iq_phase_imbalance_deg"), (-3.0, 3.0)),
        multipath_enabled=bool(raw.get("multipath_enabled", False)),
        multipath_max_delay_samples=int(raw.get("multipath_max_delay_samples", 8)),
        multipath_taps=_int_pair(raw.get("multipath_taps"), (2, 4)),
        multipath_decay=_float_pair(raw.get("multipath_decay"), (0.15, 0.55)),
    )


def apply_channel_impairments(
    iq: np.ndarray,
    sample_rate: float,
    rng: np.random.Generator | None = None,
    params: ChannelImpairmentParams | None = None,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    rng = rng or np.random.default_rng()
    params = params or ChannelImpairmentParams()
    x = np.asarray(iq, dtype=np.complex64)
    if not params.enabled:
        return x.copy(), {"enabled": False}

    time_offset = _draw_int(rng, params.time_offset_samples)
    sample_rate_offset_ppm = _draw_uniform(rng, params.sample_rate_offset_ppm)
    freq_offset_hz = _draw_uniform(rng, params.carrier_freq_offset_hz)
    phase_offset_rad = _draw_uniform(rng, params.carrier_phase_offset_rad)
    gain_db = _draw_uniform(rng, params.gain_db)
    dc_i = _draw_uniform(rng, params.dc_offset)
    dc_q = _draw_uniform(rng, params.dc_offset)
    iq_gain_db = _draw_uniform(rng, params.iq_gain_imbalance_db)
    iq_phase_deg = _draw_uniform(rng, params.iq_phase_imbalance_deg)

    if time_offset:
        x = np.roll(x, time_offset)
    x = _apply_sample_rate_offset(x, sample_rate_offset_ppm)
    if params.multipath_enabled:
        x, multipath_meta = _apply_multipath(x, rng, params)
    else:
        multipath_meta = {"multipath_enabled": False, "multipath_tap_count": 1, "multipath_max_delay": 0}

    n = np.arange(len(x), dtype=np.float32)
    phase = 2.0 * np.pi * float(freq_offset_hz) * n / float(sample_rate) + float(phase_offset_rad)
    x = x * np.exp(1j * phase).astype(np.complex64)
    x = x * np.float32(10.0 ** (float(gain_db) / 20.0))
    x = _apply_iq_imbalance(x, iq_gain_db, iq_phase_deg)
    x = x + np.complex64(dc_i + 1j * dc_q)

    meta: dict[str, float | int | bool] = {
        "enabled": True,
        "time_offset_samples": int(time_offset),
        "sample_rate_offset_ppm": float(sample_rate_offset_ppm),
        "carrier_freq_offset_hz": float(freq_offset_hz),
        "carrier_phase_offset_rad": float(phase_offset_rad),
        "gain_db": float(gain_db),
        "dc_i": float(dc_i),
        "dc_q": float(dc_q),
        "iq_gain_imbalance_db": float(iq_gain_db),
        "iq_phase_imbalance_deg": float(iq_phase_deg),
    }
    meta.update(multipath_meta)
    return x.astype(np.complex64), meta


def _apply_sample_rate_offset(iq: np.ndarray, ppm: float) -> np.ndarray:
    if abs(ppm) < 1e-12 or len(iq) < 2:
        return iq
    ratio = 1.0 + float(ppm) * 1e-6
    src = np.arange(len(iq), dtype=np.float64) * ratio
    src = np.clip(src, 0.0, len(iq) - 1.0)
    idx = np.arange(len(iq), dtype=np.float64)
    real = np.interp(src, idx, iq.real)
    imag = np.interp(src, idx, iq.imag)
    return (real + 1j * imag).astype(np.complex64)


def _apply_iq_imbalance(iq: np.ndarray, gain_db: float, phase_deg: float) -> np.ndarray:
    gain = 10.0 ** (float(gain_db) / 20.0)
    phase = np.deg2rad(float(phase_deg))
    i = iq.real * gain
    q = iq.imag * np.cos(phase) + iq.real * np.sin(phase)
    return (i + 1j * q).astype(np.complex64)


def _apply_multipath(
    iq: np.ndarray,
    rng: np.random.Generator,
    params: ChannelImpairmentParams,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    max_delay = max(0, int(params.multipath_max_delay_samples))
    min_taps, max_taps = params.multipath_taps
    tap_count = int(rng.integers(max(1, min_taps), max(max_taps, min_taps) + 1))
    delays = np.sort(rng.choice(np.arange(max_delay + 1), size=min(tap_count, max_delay + 1), replace=False))
    if len(delays) == 0 or delays[0] != 0:
        delays = np.insert(delays, 0, 0)

    taps = np.zeros(int(delays[-1]) + 1, dtype=np.complex64)
    taps[0] = 1.0 + 0.0j
    for delay in delays[1:]:
        mag = _draw_uniform(rng, params.multipath_decay) * np.exp(-delay / max(max_delay, 1))
        phase = rng.uniform(0.0, 2.0 * np.pi)
        taps[int(delay)] = mag * np.exp(1j * phase)

    y = np.convolve(iq, taps, mode="same")
    power = np.sqrt(np.sum(np.abs(taps) ** 2))
    y = y / (power + 1e-8)
    return y.astype(np.complex64), {
        "multipath_enabled": True,
        "multipath_tap_count": int(len(delays)),
        "multipath_max_delay": int(delays[-1]),
    }


def _draw_uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    return float(rng.uniform(min(lo, hi), max(lo, hi)))


def _draw_int(rng: np.random.Generator, bounds: tuple[int, int]) -> int:
    lo, hi = bounds
    return int(rng.integers(min(lo, hi), max(lo, hi) + 1))


def _float_pair(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        value = [-float(value), float(value)]
    if len(value) != 2:
        raise ValueError(f"Expected a two-value range, got {value}")
    return float(value[0]), float(value[1])


def _int_pair(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, int):
        value = [-value, value]
    if len(value) != 2:
        raise ValueError(f"Expected a two-value range, got {value}")
    return int(value[0]), int(value[1])
