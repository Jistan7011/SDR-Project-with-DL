from __future__ import annotations

import numpy as np

from src.signal.demod import recover_payload
from src.signal.frame import bits_to_text, make_frame, parse_frame, text_to_bits
from src.signal.modulate import modulate_bits
from src.signal.recovery_metrics import bit_error_rate, character_error_rate, failure_stage, packet_success


def test_text_bits_round_trip() -> None:
    bits = text_to_bits("P")
    assert bits_to_text(bits) == "P"


def test_frame_crc_pass_and_fail() -> None:
    frame = make_frame("A")
    parsed = parse_frame(frame)
    assert parsed["payload"] == "A"
    assert parsed["crc_ok"] is True
    broken = frame.copy()
    broken[-1] ^= 1
    assert parse_frame(broken)["crc_ok"] is False


def test_clean_simulation_oracle_demod_recovers_payloads() -> None:
    sample_rate = 160_000.0
    symbol_rate = 5_000.0
    sps = int(sample_rate / symbol_rate)
    cases = [("BASK", "A"), ("BFSK", "F"), ("BPSK", "P")]
    for modulation, payload in cases:
        bits = make_frame(payload)
        iq = modulate_bits(modulation, bits, sps, sample_rate)
        t = np.arange(len(iq), dtype=np.float32) / sample_rate
        channelized = iq * np.exp(-1j * 2.0 * np.pi * 100_000.0 * t)
        recovered = recover_payload(modulation, channelized, sps, sample_rate, carrier_freq=0.0)
        assert recovered["payload"] == payload
        assert recovered["crc_ok"] is True


def test_recovery_metric_helpers() -> None:
    assert bit_error_rate(np.array([1, 0, 1]), np.array([1, 1, 1])) == 1 / 3
    assert character_error_rate("A", "B") == 1.0
    assert packet_success("A", "A", True) is True
    assert failure_stage("BPSK", "BASK", "", False, -1, "P") == "classification"
    assert failure_stage("BPSK", "BPSK", "", False, -1, "P") == "frame_sync"
    assert failure_stage("BPSK", "BPSK", "P", False, 24, "P") == "crc"
    assert failure_stage("BPSK", "BPSK", "P", True, 24, "P") == "success"
