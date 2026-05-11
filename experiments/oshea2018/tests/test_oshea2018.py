from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.dataset.generate_oshea2018_synthetic import build_impairment_schedule
from src.analysis.hos_xgboost_baseline import feature_vector
from src.dataset.iq_dataset import IQDataset
from src.dataset.import_oshea2018_ota_windows import capture_quality, detect_active_region, window_indices
from src.experiment.run_oshea2018_capture_session import stable_payload_seed
from src.models.cnn1d import build_model
from src.signal.oshea2018_waveform import (
    apply_channel_impairments,
    generate_clean_modulation,
    random_bits,
    to_unit_variance_iq,
)
from src.sdr.hackrf_tx_oshea2018 import build_tx_iq, complex64_to_cs8


def test_oshea2018_waveform_to_raw_iq_shape() -> None:
    rng = np.random.default_rng(42)
    bits = random_bits(rng, 256)
    clean = generate_clean_modulation("BPSK", bits, sample_rate=2.4e6, symbol_rate=10e3)
    impaired = apply_channel_impairments(
        clean,
        rng,
        sample_rate=2.4e6,
        snr_db=20.0,
        cfo_hz=100.0,
        clock_offset=0.0,
        phase_offset=0.5,
        gain=1.0,
        multipath_taps=4,
        multipath_delay_spread=1.0,
    )
    channels = to_unit_variance_iq(impaired, 1024)
    assert channels.shape == (2, 1024)
    assert np.isfinite(channels).all()
    assert 0.8 < float(np.std(channels[0] + 1j * channels[1])) < 1.2


def test_oshea2018_models_forward() -> None:
    x = torch.randn(4, 2, 1024)
    for model_type in ["oshea2018_vgg1d", "oshea2018_resnet1d"]:
        model = build_model(model_type, input_channels=2, num_classes=3, dropout=0.1)
        y = model(x)
        assert y.shape == (4, 3)


def test_oshea2018_raw_iq_dataset_mode(tmp_path: Path) -> None:
    split = tmp_path / "train"
    split.mkdir()
    rng = np.random.default_rng(7)
    iq = rng.normal(size=(2, 1024)).astype(np.float32)
    np.savez_compressed(split / "sample.npz", iq=iq, modulation=np.asarray("BASK"), session_id=np.asarray("session_001"))
    ds = IQDataset(tmp_path, "train", feature_mode="iq")
    x, y = ds[0]
    assert x.shape == (2, 1024)
    assert int(y) == 0
    assert torch.isfinite(x).all()

def test_hos_feature_vector_is_finite() -> None:
    rng = np.random.default_rng(7)
    channels = rng.normal(size=(2, 1024)).astype(np.float32)
    features = feature_vector(channels)
    assert features.ndim == 1
    assert features.shape[0] >= 28
    assert np.isfinite(features).all()


def test_impairment_schedule_is_reusable_across_classes() -> None:
    rng = np.random.default_rng(123)
    schedule = build_impairment_schedule(
        rng,
        count=5,
        snr_values=[-5, 0, 5],
        cfo_max=1000.0,
        clock_sigma=0.001,
        gain_range=(0.5, 1.5),
        rolloff_range=(0.1, 0.4),
        multipath_spreads=[0.0, 1.0],
    )
    assert len(schedule) == 5
    for item in schedule:
        assert set(item) == {"snr_db", "cfo_hz", "clock_offset", "phase_offset", "gain", "rolloff", "multipath_delay_spread"}
        assert -1000.0 <= item["cfo_hz"] <= 1000.0


def test_window_indices_unlimited_uses_entire_region() -> None:
    indices = window_indices(start_index=1000, end_index=5000, window_len=1024, stride=512, limit=0, mode="uniform_tx_region")
    assert indices[0] == 1000
    assert indices[-1] + 1024 <= 5000
    assert len(indices) > 1


def test_active_region_detection_skips_rx_lead_noise() -> None:
    sample_rate = 1000.0
    rng = np.random.default_rng(123)
    noise = (rng.normal(scale=0.01, size=600) + 1j * rng.normal(scale=0.01, size=600)).astype(np.complex64)
    signal = (rng.normal(scale=0.10, size=1400) + 1j * rng.normal(scale=0.10, size=1400)).astype(np.complex64)
    iq = np.concatenate([noise, signal])
    result = detect_active_region(
        iq,
        sample_rate=sample_rate,
        window_len=64,
        rx_lead_seconds=0.5,
        fallback_start=700,
        fallback_end=len(iq),
        cfg={"enabled": True, "block_ms": 50, "hop_ms": 25, "min_rms_ratio": 2.0, "pad_seconds": 0.0},
    )
    assert result["active_region_found"] is True
    assert int(result["active_start_sample"]) >= 500
    assert int(result["active_end_sample"]) <= len(iq)


def test_capture_quality_rejects_noise_only_capture() -> None:
    rng = np.random.default_rng(99)
    iq = (rng.normal(scale=0.01, size=2000) + 1j * rng.normal(scale=0.01, size=2000)).astype(np.complex64)
    quality = capture_quality(
        iq,
        sample_rate=1000.0,
        start_index=700,
        end_index=1900,
        rx_lead_seconds=0.5,
        cfg={"min_tx_to_noise_rms_ratio": 1.5, "min_spectral_peak_prominence": 0.0, "max_clipping_rate": 0.02},
    )
    assert quality["quality_pass"] is False


def test_payload_seed_is_shared_across_modulations_by_capture() -> None:
    seed0 = stable_payload_seed("session_001", 0)
    seed1 = stable_payload_seed("session_001", 1)
    same_capture_seed = stable_payload_seed("session_001", 0)
    assert isinstance(seed0, int)
    assert seed0 == same_capture_seed
    assert seed0 != seed1


def test_payload_seed_changes_by_session() -> None:
    assert stable_payload_seed("session_001", 0) != stable_payload_seed("session_002", 0)


def test_oshea2018_tx_iq_uses_random_seed_and_offset() -> None:
    cfg = {
        "sdr": {
            "tx_sample_rate": 2.4e6,
            "symbol_rate": 5e3,
            "baseband_offset_hz": 500_000,
        },
        "modulation": {"bfsk_freq_dev_hz": 50_000},
    }
    iq0, sample_rate, offset = build_tx_iq("BPSK", cfg, seconds=0.1, seed=1, baseband_offset_hz=250_000)
    iq1, _, _ = build_tx_iq("BPSK", cfg, seconds=0.1, seed=2, baseband_offset_hz=250_000)
    assert iq0.dtype == np.complex64
    assert sample_rate == 2.4e6
    assert offset == 250_000
    assert len(iq0) >= 256
    assert not np.allclose(iq0, iq1)


def test_complex64_to_cs8_interleaves_iq_samples() -> None:
    iq = np.asarray([0.5 + 0.25j, -0.5 - 0.25j], dtype=np.complex64)
    cs8 = complex64_to_cs8(iq)
    assert cs8.dtype == np.int8
    assert cs8.shape == (4,)
    assert cs8[0] > 0
    assert cs8[1] > 0
    assert cs8[2] < 0
    assert cs8[3] < 0
