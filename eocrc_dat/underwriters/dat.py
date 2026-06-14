from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from eocrc_dat.underwriters.ft_transformer import FTConfig, FTTransformerEncoder
from eocrc_dat.underwriters.tabnet import TabNetConfig, TabNetEncoder


@dataclass(frozen=True)
class DATConfig:
    n_features: int
    k_steps: int = 5
    gamma: float = 1.3
    tabnet_dim: int = 64
    n_glu: int = 2
    q_select: int = 32
    ft_dim: int = 192
    ft_heads: int = 8
    ft_layers: int = 3
    ff_mult: int = 2
    dropout: float = 0.1


@dataclass
class DATTrace:
    selected: torch.Tensor
    attention: torch.Tensor


class DualAttentionTransformer(nn.Module):
    def __init__(self, cfg: DATConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.q = min(cfg.q_select, cfg.n_features)
        self.tabnet = TabNetEncoder(
            TabNetConfig(
                n_features=cfg.n_features,
                n_steps=cfg.k_steps,
                gamma=cfg.gamma,
                feature_dim=cfg.tabnet_dim,
                n_glu=cfg.n_glu,
            )
        )
        self.ft = FTTransformerEncoder(
            FTConfig(
                n_features=cfg.n_features,
                dim=cfg.ft_dim,
                heads=cfg.ft_heads,
                layers=cfg.ft_layers,
                ff_mult=cfg.ff_mult,
                dropout=cfg.dropout,
            )
        )
        self.head = nn.Linear(cfg.ft_dim, 1)

    def _gate(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.tabnet(x)
        importance = out.importance
        gate = x.shape[1] * torch.softmax(importance, dim=-1)
        return x * gate, importance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated, _ = self._gate(x)
        cls, _ = self.ft(gated)
        return self.head(cls).squeeze(-1)

    def trace(self, x: torch.Tensor) -> DATTrace:
        gated, importance = self._gate(x)
        _, attention = self.ft(gated, return_attention=True)
        if attention is None:
            raise RuntimeError("attention trace unavailable")
        ranking = importance.mean(dim=0)
        index = torch.sort(torch.topk(ranking, self.q).indices).values
        return DATTrace(selected=index, attention=attention)
