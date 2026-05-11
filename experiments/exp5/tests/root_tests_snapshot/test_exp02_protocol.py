from __future__ import annotations

import numpy as np

from src.dataset.import_exp02_sessions import assign_session_splits, validate_session_split
from src.models.cnn1d import build_model
from src.signal.channelize import channelize_and_downsample, estimate_snr_db


def test_exp02_session_split_keeps_test_at_three_sessions():
    session_ids = [f"session_{i:03d}" for i in range(1, 16)]
    split = assign_session_splits(session_ids)
    assert list(split.values()).count("train") == 9
    assert list(split.values()).count("val") == 3
    assert list(split.values()).count("test") == 3
    assert split["session_001"] == "train"
    assert split["session_015"] == "test"


def test_validate_session_split_rejects_payload_pool_mismatch():
    manifest = [
        {"session_id": "session_001", "split": "train", "modulation": "BASK", "payload": "A"},
        {"session_id": "session_001", "split": "train", "modulation": "BFSK", "payload": "A"},
        {"session_id": "session_001", "split": "train", "modulation": "BPSK", "payload": "F"},
    ]
    try:
        validate_session_split(manifest)
    except SystemExit as exc:
        assert "Payload pools differ" in str(exc)
    else:
        raise AssertionError("payload pool mismatch should fail")


def test_snr_estimator_uses_noise_only_and_active_power():
    rng = np.random.default_rng(7)
    noise = (rng.normal(scale=0.1, size=4096) + 1j * rng.normal(scale=0.1, size=4096)).astype(np.complex64)
    signal = np.ones(4096, dtype=np.complex64) * 0.5
    active = noise + signal
    result = estimate_snr_db(noise, active)
    assert result["active_power"] > result["noise_power"]
    assert result["estimated_snr_db"] > 0


def test_channelize_downsample_outputs_expected_rate_and_dtype():
    sample_rate = 2_400_000.0
    target = 160_000.0
    t = np.arange(24_000) / sample_rate
    iq = np.exp(1j * 2.0 * np.pi * 250_000.0 * t).astype(np.complex64)
    out, effective = channelize_and_downsample(iq, sample_rate, 250_000.0, 100_000.0, target)
    assert out.dtype == np.complex64
    assert abs(effective - target) < 1.0
    assert 1500 <= len(out) <= 1700


def test_model_builder_supports_exp02_candidates():
    for model_type in ("cnn1d", "vgg1d", "resnet1d"):
        model = build_model(model_type)
        assert model is not None
