from __future__ import annotations

import torch
from torch import nn


class CNN1DClassifier(nn.Module):
    def __init__(self, input_channels: int = 2, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


class VGG1DClassifier(nn.Module):
    def __init__(self, input_channels: int = 2, num_classes: int = 3, dropout: float = 0.35):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(input_channels, 32),
            conv_block(32, 32),
            nn.MaxPool1d(2),
            conv_block(32, 64),
            conv_block(64, 64),
            nn.MaxPool1d(2),
            conv_block(64, 128),
            conv_block(128, 128),
            nn.MaxPool1d(2),
            conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride), nn.BatchNorm1d(out_channels))
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.main(x) + self.skip(x))


class ResNet1DClassifier(nn.Module):
    def __init__(self, input_channels: int = 2, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.stem = conv_block(input_channels, 32, kernel_size=7)
        self.features = nn.Sequential(
            ResidualBlock1D(32, 32),
            ResidualBlock1D(32, 64, stride=2),
            ResidualBlock1D(64, 64),
            ResidualBlock1D(64, 128, stride=2),
            ResidualBlock1D(128, 128),
            ResidualBlock1D(128, 256, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(self.stem(x))))


def conv_block(in_channels: int, out_channels: int, kernel_size: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
    )


def build_model(model_type: str, input_channels: int = 2, num_classes: int = 3, dropout: float = 0.3) -> nn.Module:
    key = model_type.lower().replace("-", "_")
    if key in {"cnn1d", "baseline", "current"}:
        return CNN1DClassifier(input_channels=input_channels, num_classes=num_classes, dropout=dropout)
    if key in {"vgg", "vgg1d", "vgg_style"}:
        return VGG1DClassifier(input_channels=input_channels, num_classes=num_classes, dropout=dropout)
    if key in {"resnet", "resnet1d"}:
        return ResNet1DClassifier(input_channels=input_channels, num_classes=num_classes, dropout=dropout)
    raise ValueError(f"Unsupported model type: {model_type}")
