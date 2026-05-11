from __future__ import annotations

import numpy as np
import torch

from src.models.cnn1d import build_model
from src.train.train_exp08_multitask_classifier import compute_exp08_metrics, exp08_loss, supervised_contrastive_loss


def test_multitask_resnet_outputs_expected_heads() -> None:
    model = build_model("multitask_resnet1d", input_channels=5, num_classes=3, dropout=0.1)
    x = torch.randn(4, 5, 2048)
    outputs = model(x)
    assert outputs["logits"].shape == (4, 3)
    assert outputs["bask_binary_logits"].shape == (4, 2)
    assert outputs["bfsk_bpsk_logits"].shape == (4, 2)
    assert outputs["embedding"].shape == (4, 256)


def test_exp08_loss_masks_bask_for_bfsk_bpsk_head() -> None:
    outputs = {
        "logits": torch.randn(3, 3, requires_grad=True),
        "bask_binary_logits": torch.randn(3, 2, requires_grad=True),
        "bfsk_bpsk_logits": torch.randn(3, 2, requires_grad=True),
        "embedding": torch.randn(3, 256, requires_grad=True),
    }
    labels = torch.tensor([0, 1, 2], dtype=torch.long)
    loss = exp08_loss(outputs, labels, {"multiclass_weight": 1.0, "bask_binary_weight": 0.45, "bfsk_bpsk_weight": 0.55}, 0.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert outputs["bfsk_bpsk_logits"].grad is not None
    assert torch.allclose(outputs["bfsk_bpsk_logits"].grad[0], torch.zeros(2))


def test_supervised_contrastive_loss_is_finite() -> None:
    embedding = torch.randn(6, 16)
    labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    loss = supervised_contrastive_loss(embedding, labels)
    assert torch.isfinite(loss)


def test_exp08_boundary_metrics() -> None:
    true = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    pred = np.array([0, 1, 0, 1, 0, 2], dtype=np.int64)
    metrics = compute_exp08_metrics(true, pred)
    assert metrics["bask_to_nonbask_rate"] == 0.5
    assert metrics["bfsk_bpsk_to_bask_rate"] == 0.5
    assert metrics["worst_recall"] == 0.5
