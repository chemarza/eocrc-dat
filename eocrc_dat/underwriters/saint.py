from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from eocrc_dat.underwriters.ft_transformer import FeatureTokenizer


@dataclass(frozen=True)
class SaintConfig:
    n_features: int
    dim: int = 96
    heads: int = 8
    layers: int = 2
    dropout: float = 0.1


class _ColumnAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        attended, _ = self.attn(h, h, h, need_weights=False)
        return x + attended


class _IntersampleAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        across = self.norm(x).transpose(0, 1)
        attended, _ = self.attn(across, across, across, need_weights=False)
        return x + attended.transpose(0, 1)


class SaintClassifier(nn.Module):
    def __init__(self, cfg: SaintConfig) -> None:
        super().__init__()
        self.tokenizer = FeatureTokenizer(cfg.n_features, cfg.dim)
        self.columns = nn.ModuleList(
            [_ColumnAttention(cfg.dim, cfg.heads, cfg.dropout) for _ in range(cfg.layers)]
        )
        self.intersample = nn.ModuleList(
            [_IntersampleAttention(cfg.dim, cfg.heads, cfg.dropout) for _ in range(cfg.layers)]
        )
        self.norm = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        for column, inter in zip(self.columns, self.intersample, strict=True):
            tokens = inter(column(tokens))
        cls = self.norm(tokens)[:, 0]
        return self.head(cls).squeeze(-1)
