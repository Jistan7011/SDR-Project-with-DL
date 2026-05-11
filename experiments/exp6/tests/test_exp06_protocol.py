from __future__ import annotations

import numpy as np
import torch

from src.app.evaluate_two_stage_classifier import bfsk_bpsk_to_bask_rate
from src.dataset.binary_iq_dataset import BinaryIQDataset
from src.dataset.make_exp06_binary_datasets import STAGE1_CLASSES, STAGE2_CLASSES
from src.models.cnn1d import build_model


def test_exp06_binary_class_mappings() -> None:
    assert STAGE1_CLASSES == ["BASK", "NON_BASK"]
    assert STAGE2_CLASSES == ["BFSK", "BPSK"]


def test_binary_dataset_reads_label_field(tmp_path) -> None:
    split_dir = tmp_path / "train"
    split_dir.mkdir(parents=True)
    iq = np.zeros((2, 2048), dtype=np.float32)
    np.savez(split_dir / "sample.npz", iq=iq, modulation="BFSK", original_modulation="BFSK", label=1, label_name="NON_BASK")
    sample, label = BinaryIQDataset(tmp_path, "train", feature_mode="iq_mag_ifreq_dphase")[0]
    assert tuple(sample.shape) == (5, 2048)
    assert int(label) == 1


def test_two_stage_final_label_logic_shape() -> None:
    stage1 = build_model("resnet1d", input_channels=5, num_classes=2)
    stage2 = build_model("resnet1d", input_channels=5, num_classes=2)
    x = torch.randn(3, 5, 2048)
    assert tuple(stage1(x).shape) == (3, 2)
    assert tuple(stage2(x).shape) == (3, 2)


def test_bfsk_bpsk_to_bask_rate() -> None:
    cm = np.array([[10, 0, 0], [2, 8, 0], [3, 0, 7]], dtype=np.int64)
    assert bfsk_bpsk_to_bask_rate(cm) == 5 / 20
