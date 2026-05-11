from __future__ import annotations

import numpy as np

from src.signal.awgn import add_awgn, measure_snr
from src.signal.demod import recover_payload
from src.signal.frame import bits_to_text, make_frame, parse_frame, text_to_bits
from src.signal.modulate import modulate_bits


def test_text_bits_roundtrip():
    assert bits_to_text(text_to_bits("AFP")) == "AFP"


def test_frame_parse_roundtrip():
    result = parse_frame(make_frame("A"))
    assert result["payload"] == "A"
    assert result["crc_ok"] is True


def test_modulators_and_awgn():
    bits = make_frame("P")
    for mod in ["BASK", "BFSK", "BPSK"]:
        iq = modulate_bits(mod, bits, 16, 16000)
        assert iq.dtype == np.complex64
        assert len(iq) == len(bits) * 16
        noisy = add_awgn(iq, 10, np.random.default_rng(42))
        assert abs(measure_snr(iq, noisy) - 10) < 1.5


def test_clean_demod_payload_recovery():
    sample_rate = 2_400_000
    symbol_rate = 5_000
    sps = sample_rate // symbol_rate
    for mod, payload in [("BASK", "A"), ("BFSK", "F"), ("BPSK", "P")]:
        iq = modulate_bits(mod, make_frame(payload), sps, sample_rate)
        result = recover_payload(mod, iq, sps, sample_rate)
        assert result["payload"] == payload
        assert result["crc_ok"] is True
