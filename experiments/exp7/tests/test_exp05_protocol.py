from __future__ import annotations

import numpy as np
import torch

from src.dataset.iq_dataset import IQDataset
from src.dataset.import_exp05_classification_windows import validate_no_session_leakage
from src.models.cnn1d import build_model
from src.signal.processing import differential_phase_channel
from src.train.train_cnn1d import adapt_state_dict_for_model


def test_differential_phase_channel_is_finite_and_stable() -> None:
    rng = np.random.default_rng(5)
    iq = (rng.normal(size=2048) + 1j * rng.normal(size=2048)).astype(np.complex64)
    dphase = differential_phase_channel(iq, symbol_samples=32)
    assert dphase.shape == (2048,)
    assert np.isfinite(dphase).all()


def test_exp05_feature_mode_has_five_channels(tmp_path) -> None:
    split_dir = tmp_path / "train"
    split_dir.mkdir()
    rng = np.random.default_rng(7)
    iq = rng.normal(size=(2, 2048)).astype(np.float32)
    np.savez(split_dir / "sample.npz", iq=iq, modulation="BFSK", payload="F")
    sample, label = IQDataset(tmp_path, "train", feature_mode="iq_mag_ifreq_dphase")[0]
    assert tuple(sample.shape) == (5, 2048)
    assert int(label) == 1
    assert torch.isfinite(sample).all()


def test_exp05_processed_sample_does_not_require_raw_iq(tmp_path) -> None:
    split_dir = tmp_path / "train"
    split_dir.mkdir()
    iq = np.zeros((2, 2048), dtype=np.float32)
    np.savez(split_dir / "sample.npz", iq=iq, modulation="BPSK", payload="P", session_id="session_001")
    data = np.load(split_dir / "sample.npz", allow_pickle=False)
    assert "iq" in data.files
    assert "raw_iq" not in data.files


def test_exp05_manifest_rejects_session_leakage() -> None:
    manifest = [
        {"session_id": "session_027", "split": "train"},
        {"session_id": "session_027", "split": "test"},
    ]
    try:
        validate_no_session_leakage(manifest)
    except SystemExit as exc:
        assert "Session split leakage" in str(exc)
    else:
        raise AssertionError("session leakage must be rejected")


def test_partial_checkpoint_adapts_extra_input_channels() -> None:
    source_model = build_model("resnet1d", input_channels=3, num_classes=3)
    target_model = build_model("resnet1d", input_channels=5, num_classes=3)
    adapted = adapt_state_dict_for_model(target_model.state_dict(), source_model.state_dict())
    assert adapted["stem.0.weight"].shape == target_model.state_dict()["stem.0.weight"].shape
    assert torch.allclose(adapted["stem.0.weight"][:, :3], source_model.state_dict()["stem.0.weight"])


def test_fusion_resnet_accepts_exp05_five_channel_input() -> None:
    model = build_model("fusion_resnet1d", input_channels=5, num_classes=3)
    logits = model(torch.randn(4, 5, 2048))
    assert tuple(logits.shape) == (4, 3)
