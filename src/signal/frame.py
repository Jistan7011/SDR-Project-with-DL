from __future__ import annotations

import numpy as np


DEFAULT_PREAMBLE = "1010101010101010"
DEFAULT_SYNC_WORD = "11001100"


def text_to_bits(text: str) -> np.ndarray:
    bits: list[int] = []
    for ch in text:
        byte = ord(ch)
        bits.extend((byte >> i) & 1 for i in range(7, -1, -1))
    return np.asarray(bits, dtype=np.uint8)


def bits_to_text(bits: np.ndarray) -> str:
    clean = np.asarray(bits, dtype=np.uint8).ravel()
    chars: list[str] = []
    for start in range(0, len(clean) - 7, 8):
        byte = 0
        for bit in clean[start : start + 8]:
            byte = (byte << 1) | int(bit)
        chars.append(chr(byte))
    return "".join(chars)


def bits_from_string(bit_string: str) -> np.ndarray:
    return np.asarray([1 if ch == "1" else 0 for ch in bit_string], dtype=np.uint8)


def crc8(bits: np.ndarray, poly: int = 0x07, init: int = 0x00) -> int:
    crc = init
    for bit in np.asarray(bits, dtype=np.uint8).ravel():
        crc ^= int(bit) << 7
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def make_frame(
    payload: str,
    preamble: str = DEFAULT_PREAMBLE,
    sync_word: str = DEFAULT_SYNC_WORD,
) -> np.ndarray:
    payload_bits = text_to_bits(payload)
    return make_frame_from_bits(payload_bits, preamble=preamble, sync_word=sync_word)


def make_frame_from_bits(
    payload_bits: np.ndarray,
    preamble: str = DEFAULT_PREAMBLE,
    sync_word: str = DEFAULT_SYNC_WORD,
) -> np.ndarray:
    payload_bits = np.asarray(payload_bits, dtype=np.uint8).ravel()
    crc_bits = np.asarray([(crc8(payload_bits) >> i) & 1 for i in range(7, -1, -1)], dtype=np.uint8)
    return np.concatenate([bits_from_string(preamble), bits_from_string(sync_word), payload_bits, crc_bits])


def find_pattern(bits: np.ndarray, pattern: np.ndarray, max_errors: int = 0) -> int:
    data = np.asarray(bits, dtype=np.uint8).ravel()
    pat = np.asarray(pattern, dtype=np.uint8).ravel()
    if len(data) < len(pat):
        return -1
    for i in range(len(data) - len(pat) + 1):
        errors = int(np.count_nonzero(data[i : i + len(pat)] != pat))
        if errors <= max_errors:
            return i
    return -1


def parse_frame(
    bits: np.ndarray,
    payload_bytes: int = 1,
    preamble: str = DEFAULT_PREAMBLE,
    sync_word: str = DEFAULT_SYNC_WORD,
    max_sync_errors: int = 1,
) -> dict[str, object]:
    sync = np.concatenate([bits_from_string(preamble), bits_from_string(sync_word)])
    start = find_pattern(bits, sync, max_errors=max_sync_errors)
    if start < 0:
        return {"payload": "", "crc_ok": False, "bits": np.asarray(bits, dtype=np.uint8), "start": -1}
    data_start = start + len(sync)
    payload_len = payload_bytes * 8
    data = np.asarray(bits, dtype=np.uint8).ravel()
    payload_bits = data[data_start : data_start + payload_len]
    crc_bits = data[data_start + payload_len : data_start + payload_len + 8]
    if len(payload_bits) < payload_len or len(crc_bits) < 8:
        return {"payload": "", "crc_ok": False, "bits": payload_bits, "start": data_start}
    payload = bits_to_text(payload_bits)
    received_crc = int("".join(str(int(b)) for b in crc_bits), 2)
    return {
        "payload": payload,
        "crc_ok": received_crc == crc8(payload_bits),
        "bits": payload_bits,
        "start": data_start,
    }
