from __future__ import annotations

import numpy as np

from src.signal.demod_bask import demod_bask
from src.signal.demod_bfsk import demod_bfsk
from src.signal.demod_bpsk import demod_bpsk
from src.signal.frame import parse_frame


def demodulate(
    modulation: str,
    iq: np.ndarray,
    samples_per_symbol: int,
    sample_rate: float,
    carrier_freq: float = 100_000.0,
) -> np.ndarray:
    mod = modulation.upper()
    if mod == "BASK":
        return demod_bask(iq, samples_per_symbol)
    if mod == "BFSK":
        return demod_bfsk(iq, samples_per_symbol)
    if mod == "BPSK":
        return demod_bpsk(iq, samples_per_symbol, sample_rate=sample_rate, carrier_freq=carrier_freq)
    raise ValueError(f"Unsupported modulation: {modulation}")


def recover_payload(
    modulation: str,
    iq: np.ndarray,
    samples_per_symbol: int,
    sample_rate: float,
    payload_bytes: int = 1,
    carrier_freq: float = 100_000.0,
) -> dict[str, object]:
    if modulation.upper() == "BPSK":
        return recover_bpsk_payload(iq, samples_per_symbol, sample_rate, payload_bytes, carrier_freq)
    bits = demodulate(modulation, iq, samples_per_symbol, sample_rate, carrier_freq=carrier_freq)
    result = parse_frame(bits, payload_bytes=payload_bytes)
    result["recovered_bits"] = bits
    return result


def recover_bpsk_payload(
    iq: np.ndarray,
    samples_per_symbol: int,
    sample_rate: float,
    payload_bytes: int,
    carrier_freq: float,
) -> dict[str, object]:
    arr = np.asarray(iq, dtype=np.complex64)
    differential = recover_bpsk_differential(arr, samples_per_symbol, payload_bytes)
    if differential.get("crc_ok", False):
        return differential
    best: dict[str, object] | None = None
    best_score = -1
    for phase in np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False):
        rotated = arr * np.exp(-1j * phase)
        bits = demodulate("BPSK", rotated, samples_per_symbol, sample_rate, carrier_freq=carrier_freq)
        for candidate_bits in (bits, 1 - bits):
            result = parse_frame(candidate_bits, payload_bytes=payload_bytes)
            result["recovered_bits"] = candidate_bits
            result["phase_correction_rad"] = float(phase)
            score = score_frame_result(result)
            if score > best_score:
                best = result
                best_score = score
            if result.get("crc_ok", False):
                return result
    return best or {"payload": "", "crc_ok": False, "bits": np.asarray([], dtype=np.uint8), "recovered_bits": np.asarray([], dtype=np.uint8), "start": -1}


def recover_bpsk_differential(iq: np.ndarray, samples_per_symbol: int, payload_bytes: int) -> dict[str, object]:
    arr = np.asarray(iq, dtype=np.complex64)
    n_symbols = len(arr) // samples_per_symbol
    if n_symbols < 2:
        return {"payload": "", "crc_ok": False, "bits": np.asarray([], dtype=np.uint8), "recovered_bits": np.asarray([], dtype=np.uint8), "start": -1}
    symbols = arr[: n_symbols * samples_per_symbol].reshape(n_symbols, samples_per_symbol).mean(axis=1)
    transitions = (np.real(symbols[1:] * np.conj(symbols[:-1])) < 0.0).astype(np.uint8)
    best: dict[str, object] | None = None
    best_score = -1
    for seed in (0, 1):
        bits = np.empty(n_symbols, dtype=np.uint8)
        bits[0] = seed
        for index, transition in enumerate(transitions, start=1):
            bits[index] = bits[index - 1] ^ int(transition)
        for candidate_bits in (bits, 1 - bits):
            result = parse_frame(candidate_bits, payload_bytes=payload_bytes)
            result["recovered_bits"] = candidate_bits
            result["bpsk_recovery_mode"] = "differential"
            score = score_frame_result(result)
            if score > best_score:
                best = result
                best_score = score
            if result.get("crc_ok", False):
                return result
    return best or {"payload": "", "crc_ok": False, "bits": np.asarray([], dtype=np.uint8), "recovered_bits": np.asarray([], dtype=np.uint8), "start": -1}


def score_frame_result(result: dict[str, object]) -> int:
    if bool(result.get("crc_ok", False)):
        return 3
    if int(result.get("start", -1)) >= 0 and str(result.get("payload", "")):
        return 2
    if int(result.get("start", -1)) >= 0:
        return 1
    return 0
