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


class MultiTaskResNet1DClassifier(nn.Module):
    def __init__(self, input_channels: int = 5, num_classes: int = 3, dropout: float = 0.35):
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
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, num_classes)
        self.bask_binary_head = nn.Linear(256, 2)
        self.bfsk_bpsk_head = nn.Linear(256, 2)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(self.stem(x))).flatten(1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.encode(x)
        z = self.dropout(embedding)
        return {
            "logits": self.classifier(z),
            "bask_binary_logits": self.bask_binary_head(z),
            "bfsk_bpsk_logits": self.bfsk_bpsk_head(z),
            "embedding": embedding,
        }


class FusionResNet1DClassifier(nn.Module):
    def __init__(self, input_channels: int = 4, num_classes: int = 3, dropout: float = 0.35, spectral_bins: int = 128):
        super().__init__()
        self.spectral_bins = spectral_bins
        self.time_stem = conv_block(input_channels, 32, kernel_size=7)
        self.time_features = nn.Sequential(
            ResidualBlock1D(32, 32),
            ResidualBlock1D(32, 64, stride=2),
            ResidualBlock1D(64, 64),
            ResidualBlock1D(64, 128, stride=2),
            ResidualBlock1D(128, 128),
            ResidualBlock1D(128, 256, stride=2),
        )
        self.time_pool = nn.AdaptiveAvgPool1d(1)
        self.spectral_mlp = nn.Sequential(
            nn.Linear(spectral_bins, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 + 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time_embedding = self.time_pool(self.time_features(self.time_stem(x))).flatten(1)
        complex_iq = torch.complex(x[:, 0], x[:, 1])
        spectrum = torch.fft.fftshift(torch.fft.fft(complex_iq, dim=-1), dim=-1)
        power = torch.log1p(torch.abs(spectrum) ** 2)
        power = nn.functional.adaptive_avg_pool1d(power[:, None, :], self.spectral_bins).squeeze(1)
        power = (power - power.mean(dim=1, keepdim=True)) / (power.std(dim=1, keepdim=True) + 1e-8)
        spectral_embedding = self.spectral_mlp(power.float())
        return self.classifier(torch.cat([time_embedding, spectral_embedding], dim=1))


class Exp09FusionResNet1DClassifier(nn.Module):
    def __init__(
        self,
        input_channels: int = 5,
        num_classes: int = 3,
        dropout: float = 0.35,
        spectral_bins: int = 128,
        evidence_dim: int = 5,
    ):
        super().__init__()
        self.time_stem = conv_block(input_channels, 32, kernel_size=7)
        self.time_features = nn.Sequential(
            ResidualBlock1D(32, 32),
            ResidualBlock1D(32, 64, stride=2),
            ResidualBlock1D(64, 64),
            ResidualBlock1D(64, 128, stride=2),
            ResidualBlock1D(128, 128),
            ResidualBlock1D(128, 256, stride=2),
        )
        self.time_pool = nn.AdaptiveAvgPool1d(1)
        self.spectral_mlp = nn.Sequential(
            nn.Linear(spectral_bins, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.evidence_mlp = nn.Sequential(
            nn.Linear(evidence_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(256 + 64 + 32, num_classes)
        self.bask_binary_head = nn.Linear(256 + 64 + 32, 2)
        self.bfsk_bpsk_head = nn.Linear(256 + 64 + 32, 2)

    def encode(self, inputs: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if isinstance(inputs, dict):
            time = inputs["time"]
            spectral = inputs["spectral"]
            evidence = inputs["evidence"]
        else:
            time = inputs
            complex_iq = torch.complex(time[:, 0], time[:, 1])
            spectrum = torch.fft.fftshift(torch.fft.fft(complex_iq, dim=-1), dim=-1)
            spectral = torch.log1p(torch.abs(spectrum) ** 2)
            spectral = nn.functional.adaptive_avg_pool1d(spectral[:, None, :], 128).squeeze(1)
            evidence = torch.zeros((time.shape[0], 5), dtype=time.dtype, device=time.device)
        time_embedding = self.time_pool(self.time_features(self.time_stem(time))).flatten(1)
        spectral = (spectral - spectral.mean(dim=1, keepdim=True)) / (spectral.std(dim=1, keepdim=True) + 1e-8)
        spectral_embedding = self.spectral_mlp(spectral.float())
        evidence_embedding = self.evidence_mlp(evidence.float())
        return torch.cat([time_embedding, spectral_embedding, evidence_embedding], dim=1)

    def forward(self, inputs: dict[str, torch.Tensor] | torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.encode(inputs)
        z = self.dropout(embedding)
        return {
            "logits": self.classifier(z),
            "bask_binary_logits": self.bask_binary_head(z),
            "bfsk_bpsk_logits": self.bfsk_bpsk_head(z),
            "embedding": embedding,
        }


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
    if key in {"multitask_resnet", "multitask_resnet1d", "multitask_resnet1d_margin"}:
        return MultiTaskResNet1DClassifier(input_channels=input_channels, num_classes=num_classes, dropout=dropout)
    if key in {"fusion_resnet", "fusion_resnet1d", "feature_fusion"}:
        return FusionResNet1DClassifier(input_channels=input_channels, num_classes=num_classes, dropout=dropout)
    if key in {"fusion_resnet1d_exp9", "exp9_fusion_resnet1d", "fusion_resnet1d_exp9_margin"}:
        return Exp09FusionResNet1DClassifier(input_channels=input_channels, num_classes=num_classes, dropout=dropout)
    raise ValueError(f"Unsupported model type: {model_type}")
