from __future__ import annotations

import numpy as np


def normalize_iq(iq: np.ndarray) -> np.ndarray:
    arr = np.asarray(iq, dtype=np.complex64)
    arr = arr - np.mean(arr)
    scale = np.std(arr)
    return (arr / (scale + 1e-8)).astype(np.complex64)


def iq_to_channels(iq: np.ndarray, window_size: int) -> np.ndarray:
    arr = normalize_iq(iq)
    if len(arr) < window_size:
        arr = np.pad(arr, (0, window_size - len(arr)))
    arr = arr[:window_size]
    return np.stack([arr.real, arr.imag], axis=0).astype(np.float32)


def complex64_from_channels(channels: np.ndarray) -> np.ndarray:
    x = np.asarray(channels)
    if x.shape[0] < 2:
        raise ValueError(f"Expected at least I/Q channels [2, N], got {x.shape}")
    return (x[0] + 1j * x[1]).astype(np.complex64)


def instantaneous_frequency_channel(iq: np.ndarray) -> np.ndarray:
    arr = normalize_iq(iq)
    phase_step = np.angle(arr[1:] * np.conj(arr[:-1]))
    phase_step = np.pad(phase_step, (1, 0))
    scale = np.std(phase_step)
    return (phase_step / (scale + 1e-8)).astype(np.float32)


def differential_phase_channel(iq: np.ndarray, symbol_samples: int = 32) -> np.ndarray:
    arr = normalize_iq(iq)
    delay = max(1, int(symbol_samples))
    phase_step = np.zeros(len(arr), dtype=np.float32)
    if len(arr) > delay:
        phase_step[delay:] = np.angle(arr[delay:] * np.conj(arr[:-delay])).astype(np.float32)
    return ((phase_step - np.mean(phase_step)) / (np.std(phase_step) + 1e-8)).astype(np.float32)


def magnitude_channel(iq: np.ndarray) -> np.ndarray:
    arr = normalize_iq(iq)
    magnitude = np.abs(arr).astype(np.float32)
    return ((magnitude - np.mean(magnitude)) / (np.std(magnitude) + 1e-8)).astype(np.float32)


def psd_feature(iq: np.ndarray, bins: int = 128) -> np.ndarray:
    arr = normalize_iq(iq)
    spectrum = np.fft.fftshift(np.fft.fft(arr))
    power = np.log1p(np.abs(spectrum) ** 2).astype(np.float32)
    if len(power) != bins:
        x_old = np.linspace(0.0, 1.0, num=len(power), dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, num=bins, dtype=np.float32)
        power = np.interp(x_new, x_old, power).astype(np.float32)
    return ((power - np.mean(power)) / (np.std(power) + 1e-8)).astype(np.float32)
