from __future__ import annotations

import numpy as np
import torch

from src.dataset.iq_dataset import IQDataset
from src.dataset.import_exp03_sessions import exp03_split, validate_exp03_manifest
from src.models.cnn1d import build_model


def test_exp03_split_defines_reference_and_1m_test_sets():
    assert exp03_split("session_001") == "train"
    assert exp03_split("session_012") == "val"
    assert exp03_split("session_015") == "test_a"
    assert exp03_split("session_016") == "test_b"
    assert exp03_split("session_021") == "test_b"


def test_exp03_manifest_rejects_test_b_non_exp3_session():
    manifest = [{"session_id": "session_015", "split": "test_b", "modulation": "BASK", "payload": "A"}]
    try:
        validate_exp03_manifest(manifest)
    except SystemExit as exc:
        assert "test_b contains non-exp3 sessions" in str(exc)
    else:
        raise AssertionError("test_b must contain only session_016 and later")


def test_exp03_feature_modes_have_expected_shapes(tmp_path):
    split_dir = tmp_path / "train"
    split_dir.mkdir()
    rng = np.random.default_rng(3)
    iq = rng.normal(size=(2, 1024)).astype(np.float32)
    np.savez(split_dir / "sample.npz", iq=iq, modulation="BASK", payload="A")
    expected = {
        "iq": (2, 1024),
        "iq_magnitude": (3, 1024),
        "iq_instantaneous_frequency": (3, 1024),
        "iq_magnitude_instantaneous_frequency": (4, 1024),
        "fusion": (5, 1024),
    }
    for feature_mode, shape in expected.items():
        sample, label = IQDataset(tmp_path, "train", feature_mode=feature_mode)[0]
        assert tuple(sample.shape) == shape
        assert int(label) == 0


def test_fusion_resnet_forward_shape():
    model = build_model("fusion_resnet1d", input_channels=4, num_classes=3)
    x = torch.randn(5, 4, 1024)
    logits = model(x)
    assert tuple(logits.shape) == (5, 3)
