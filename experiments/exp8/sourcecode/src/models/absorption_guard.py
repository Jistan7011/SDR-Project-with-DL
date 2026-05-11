from __future__ import annotations

import torch
from torch import nn


class AbsorptionGuardMLP(nn.Module):
    def __init__(self, input_dim: int = 11, hidden_dim: int = 48, num_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
