from __future__ import annotations

import numpy as np
import torch

from src.models.cnn1d import build_model
from src.signal.canonicalization import canonicalize_iq, estimate_phase_slope


def test_exp09_canonicalization_shapes_and_finite_values() -> None:
    rng = np.random.default_rng(42)
    iq = (rng.normal(size=2048) + 1j * rng.normal(size=2048)).astype(np.complex64)
    channels = np.stack([iq.real + 0.25, iq.imag - 0.1], axis=0).astype(np.float32)
    result = canonicalize_iq(channels, sample_rate=160000.0, spectral_bins=128)
    assert result.time_features.shape == (5, 2048)
    assert result.spectral_feature.shape == (128,)
    assert result.evidence_feature.shape == (5,)
    assert np.all(np.isfinite(result.time_features))
    assert np.all(np.isfinite(result.spectral_feature))
    assert np.all(np.isfinite(result.evidence_feature))
    assert abs(np.mean(result.iq.real)) < 1e-5
    assert abs(np.mean(result.iq.imag)) < 1e-5


def test_exp09_cfo_correction_reduces_phase_slope_on_tone() -> None:
    rng = np.random.default_rng(123)
    n = np.arange(2048, dtype=np.float32)
    symbols = rng.choice([-1.0, 1.0], size=2048).astype(np.float32)
    signal = (symbols * np.exp(1j * 0.09 * n)).astype(np.complex64)
    channels = np.stack([signal.real, signal.imag], axis=0).astype(np.float32)
    before = abs(estimate_phase_slope(signal))
    result = canonicalize_iq(channels, sample_rate=160000.0, spectral_bins=128)
    after = abs(estimate_phase_slope(result.iq))
    assert after < before * 0.5


def test_exp09_fusion_model_outputs_multitask_heads() -> None:
    model = build_model("fusion_resnet1d_exp9", input_channels=5, num_classes=3, dropout=0.1)
    outputs = model(
        {
            "time": torch.randn(4, 5, 2048),
            "spectral": torch.randn(4, 128),
            "evidence": torch.randn(4, 5),
        }
    )
    assert outputs["logits"].shape == (4, 3)
    assert outputs["bask_binary_logits"].shape == (4, 2)
    assert outputs["bfsk_bpsk_logits"].shape == (4, 2)
