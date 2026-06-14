from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class MLPConfig:
    n_features: int
    hidden: tuple[int, ...] = (256, 128)
    dropout: float = 0.1


class MLPClassifier(nn.Module):
    def __init__(self, cfg: MLPConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = cfg.n_features
        for size in cfg.hidden:
            layers.append(nn.Linear(width, size))
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(cfg.dropout))
            width = size
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
