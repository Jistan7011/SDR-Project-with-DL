from __future__ import annotations

import math

import torch
from torch import nn


class ResidualUnit1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class ResidualStack1D(nn.Module):
    def __init__(self, channels: int, depth: int = 2, kernel_size: int = 5):
        super().__init__()
        units: list[nn.Module] = [
            nn.Conv1d(channels, channels, kernel_size=kernel_size, stride=2, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        ]
        for _ in range(max(1, depth)):
            units.append(ResidualUnit1D(channels, kernel_size=kernel_size))
        self.stack = nn.Sequential(*units)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x)


class ResNet1DClassifier(nn.Module):
    """Residual-network AMC model for short IQ windows.

    The default layout follows the O'Shea et al. OTA radio-classification
    pattern for 2 x 1024 IQ input: six residual stacks reduce the time axis
    to 16 samples, followed by two SELU dense layers and a softmax head.
    """

    def __init__(
        self,
        input_channels: int = 2,
        num_classes: int = 3,
        channels: int = 32,
        stacks: int = 6,
        stack_depth: int = 2,
        kernel_size: int = 5,
        fc_units: int = 128,
        dropout: float = 0.1,
        pooled_length: int = 16,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.residual_stacks = nn.Sequential(
            *[ResidualStack1D(channels, depth=stack_depth, kernel_size=kernel_size) for _ in range(stacks)]
        )
        self.pool = nn.AdaptiveAvgPool1d(pooled_length)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * pooled_length, fc_units),
            nn.SELU(inplace=True),
            nn.AlphaDropout(dropout),
            nn.Linear(fc_units, fc_units),
            nn.SELU(inplace=True),
            nn.AlphaDropout(dropout),
            nn.Linear(fc_units, num_classes),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                if isinstance(module, nn.Conv1d):
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                else:
                    _lecun_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.residual_stacks(x)
        x = self.pool(x)
        return self.classifier(x)

    def freeze_backbone(self) -> None:
        for module in (self.stem, self.residual_stacks):
            for parameter in module.parameters():
                parameter.requires_grad = False


def _lecun_normal_(tensor: torch.Tensor) -> None:
    fan_in = nn.init._calculate_correct_fan(tensor, "fan_in")
    std = 1.0 / math.sqrt(fan_in)
    with torch.no_grad():
        tensor.normal_(0.0, std)
