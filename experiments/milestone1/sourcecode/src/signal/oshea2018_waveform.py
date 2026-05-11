from __future__ import annotations

import numpy as np

try:
    from scipy import signal
except Exception:  # pragma: no cover - SDR runtime should have scipy, fallback stays portable.
    signal = None


def random_bits(rng: np.random.Generator, count: int) -> np.ndarray:
    return rng.integers(0, 2, size=count, dtype=np.uint8)


def root_raised_cosine_taps(samples_per_symbol: int, span_symbols: int = 8, rolloff: float = 0.35) -> np.ndarray:
    sps = int(samples_per_symbol)
    n = np.arange(-span_symbols * sps, span_symbols * sps + 1, dtype=np.float64)
    t = n / float(sps)
    beta = float(rolloff)
    taps = np.zeros_like(t)
    for idx, value in enumerate(t):
        if abs(value) < 1e-12:
            taps[idx] = 1.0 - beta + 4.0 * beta / np.pi
        elif beta > 0 and abs(abs(value) - 1.0 / (4.0 * beta)) < 1e-8:
            taps[idx] = (beta / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
        else:
            numerator = np.sin(np.pi * value * (1.0 - beta)) + 4.0 * beta * value * np.cos(np.pi * value * (1.0 + beta))
            denominator = np.pi * value * (1.0 - (4.0 * beta * value) ** 2)
            taps[idx] = numerator / denominator
    taps = taps / np.sqrt(np.sum(taps * taps) + 1e-12)
    return taps.astype(np.float32)


def upsample_and_shape(symbols: np.ndarray, samples_per_symbol: int, rolloff: float) -> np.ndarray:
    taps = root_raised_cosine_taps(samples_per_symbol, rolloff=rolloff)
    symbol_arr = symbols.astype(np.complex64)
    if signal is not None:
        shaped_full = signal.upfirdn(taps.astype(np.complex64), symbol_arr, up=samples_per_symbol)
        expected = len(symbol_arr) * samples_per_symbol
        delay = (len(taps) - 1) // 2
        shaped = shaped_full[delay : delay + expected]
        if len(shaped) < expected:
            shaped = np.pad(shaped, (0, expected - len(shaped)))
        return shaped.astype(np.complex64)
    upsampled = np.zeros(len(symbol_arr) * samples_per_symbol, dtype=np.complex64)
    upsampled[::samples_per_symbol] = symbol_arr
    shaped = np.convolve(upsampled, taps.astype(np.complex64), mode="same")
    return shaped.astype(np.complex64)


def generate_clean_modulation(
    modulation: str,
    bits: np.ndarray,
    sample_rate: float,
    symbol_rate: float,
    rolloff: float = 0.35,
    bfsk_freq_dev_hz: float = 50_000.0,
    bask_low_amplitude: float = 0.15,
) -> np.ndarray:
    sps = max(1, int(round(sample_rate / symbol_rate)))
    mod = modulation.upper()
    bit_values = np.asarray(bits, dtype=np.float32)
    if mod == "BASK":
        symbols = np.where(bit_values > 0.5, 1.0, bask_low_amplitude).astype(np.complex64)
        return upsample_and_shape(symbols, sps, rolloff)
    if mod == "BPSK":
        symbols = np.where(bit_values > 0.5, -1.0, 1.0).astype(np.complex64)
        return upsample_and_shape(symbols, sps, rolloff)
    if mod == "BFSK":
        freqs = np.repeat(np.where(bit_values > 0.5, bfsk_freq_dev_hz, -bfsk_freq_dev_hz), sps)
        phase = np.cumsum(2.0 * np.pi * freqs / float(sample_rate))
        return np.exp(1j * phase).astype(np.complex64)
    raise ValueError(f"Unsupported modulation: {modulation}")


def apply_channel_impairments(
    iq: np.ndarray,
    rng: np.random.Generator,
    sample_rate: float,
    snr_db: float,
    cfo_hz: float,
    clock_offset: float,
    phase_offset: float,
    gain: float,
    multipath_taps: int,
    multipath_delay_spread: float,
) -> np.ndarray:
    impaired = np.asarray(iq, dtype=np.complex64)
    if abs(clock_offset) > 1e-10:
        old_x = np.arange(len(impaired), dtype=np.float64)
        new_x = old_x * (1.0 + float(clock_offset))
        real = np.interp(old_x, new_x, impaired.real, left=0.0, right=0.0)
        imag = np.interp(old_x, new_x, impaired.imag, left=0.0, right=0.0)
        impaired = (real + 1j * imag).astype(np.complex64)
    if multipath_taps > 1 and multipath_delay_spread > 0:
        delays = np.arange(multipath_taps, dtype=np.float32)
        envelope = np.exp(-delays / max(float(multipath_delay_spread), 1e-6))
        coeff = (rng.normal(size=multipath_taps) + 1j * rng.normal(size=multipath_taps)).astype(np.complex64)
        coeff = coeff * envelope.astype(np.complex64)
        coeff = coeff / (np.sqrt(np.sum(np.abs(coeff) ** 2)) + 1e-8)
        impaired = np.convolve(impaired, coeff, mode="same").astype(np.complex64)
    n = np.arange(len(impaired), dtype=np.float64)
    impaired = impaired * np.exp(1j * (2.0 * np.pi * float(cfo_hz) * n / float(sample_rate) + float(phase_offset)))
    impaired = (float(gain) * impaired).astype(np.complex64)
    signal_power = float(np.mean(np.abs(impaired) ** 2))
    noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0) + 1e-12)
    noise = (rng.normal(size=len(impaired)) + 1j * rng.normal(size=len(impaired))) * np.sqrt(noise_power / 2.0)
    return (impaired + noise).astype(np.complex64)


def to_unit_variance_iq(iq: np.ndarray, window_len: int) -> np.ndarray:
    arr = np.asarray(iq, dtype=np.complex64)
    if len(arr) < window_len:
        arr = np.pad(arr, (0, window_len - len(arr)))
    arr = arr[:window_len]
    arr = arr - np.mean(arr)
    arr = arr / (np.std(arr) + 1e-8)
    return np.stack([arr.real, arr.imag], axis=0).astype(np.float32)
