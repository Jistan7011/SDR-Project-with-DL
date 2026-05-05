from __future__ import annotations

from typing import Any

from torch import nn

from src.models.cnn1d import CNN1DClassifier
from src.models.resnet1d import ResNet1DClassifier


def build_model(config: dict[str, Any]) -> nn.Module:
    model_cfg = config.get("model", {})
    model_type = str(model_cfg.get("type", "cnn1d")).lower()
    common = {
        "input_channels": int(model_cfg.get("input_channels", 2)),
        "num_classes": int(model_cfg.get("num_classes", 3)),
        "dropout": float(model_cfg.get("dropout", 0.3)),
    }
    if model_type in {"cnn", "cnn1d"}:
        return CNN1DClassifier(**common)
    if model_type in {"resnet", "resnet1d", "rn"}:
        return ResNet1DClassifier(
            **common,
            channels=int(model_cfg.get("channels", 32)),
            stacks=int(model_cfg.get("stacks", 6)),
            stack_depth=int(model_cfg.get("stack_depth", 2)),
            kernel_size=int(model_cfg.get("kernel_size", 5)),
            fc_units=int(model_cfg.get("fc_units", 128)),
            pooled_length=int(model_cfg.get("pooled_length", 16)),
        )
    raise ValueError(f"Unsupported model.type: {model_type}")
