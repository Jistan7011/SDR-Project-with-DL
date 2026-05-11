from __future__ import annotations

from typing import Sequence

import numpy as np


def bit_error_rate(expected: Sequence[int] | np.ndarray, recovered: Sequence[int] | np.ndarray) -> float:
    exp = np.asarray(expected, dtype=np.uint8).reshape(-1)
    rec = np.asarray(recovered, dtype=np.uint8).reshape(-1)
    if len(exp) == 0:
        return 0.0 if len(rec) == 0 else 1.0
    n = max(len(exp), len(rec))
    mismatches = abs(len(exp) - len(rec))
    overlap = min(len(exp), len(rec))
    if overlap:
        mismatches += int(np.count_nonzero(exp[:overlap] != rec[:overlap]))
    return float(mismatches / n)


def character_error_rate(expected: str, recovered: str | None) -> float:
    rec = recovered or ""
    if not expected:
        return 0.0 if not rec else 1.0
    n = max(len(expected), len(rec))
    mismatches = abs(len(expected) - len(rec))
    overlap = min(len(expected), len(rec))
    mismatches += sum(1 for a, b in zip(expected[:overlap], rec[:overlap]) if a != b)
    return float(mismatches / n)


def packet_success(expected_payload: str, recovered_payload: str | None, crc_ok: bool) -> bool:
    return bool(crc_ok and (recovered_payload or "") == expected_payload)


def failure_stage(
    expected_modulation: str,
    predicted_modulation: str,
    recovered_payload: str | None,
    crc_ok: bool,
    frame_start: int,
    expected_payload: str,
) -> str:
    if predicted_modulation != expected_modulation:
        return "classification"
    if frame_start < 0:
        return "frame_sync"
    if recovered_payload is None or recovered_payload == "":
        return "demodulation"
    if not crc_ok:
        return "crc"
    if recovered_payload != expected_payload:
        return "demodulation"
    return "success"
