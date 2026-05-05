from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.models.factory import build_model


def test_resnet1d_output_shape():
    cfg = {
        "model": {
            "type": "resnet1d",
            "input_channels": 2,
            "num_classes": 3,
            "channels": 32,
            "stacks": 6,
            "stack_depth": 1,
            "kernel_size": 5,
            "fc_units": 128,
            "pooled_length": 16,
            "dropout": 0.1,
        }
    }
    model = build_model(cfg)
    y = model(torch.randn(4, 2, 1024))
    assert tuple(y.shape) == (4, 3)


def test_resnet1d_freeze_backbone():
    model = build_model({"model": {"type": "resnet1d", "input_channels": 2, "num_classes": 3}})
    model.freeze_backbone()
    assert not any(parameter.requires_grad for parameter in model.stem.parameters())
    assert any(parameter.requires_grad for parameter in model.classifier.parameters())
