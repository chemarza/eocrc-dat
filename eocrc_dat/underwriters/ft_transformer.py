from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class FTConfig:
    n_features: int
    dim: int = 192
    heads: int = 8
    layers: int = 3
    ff_mult: int = 2
    dropout: float = 0.1


class FeatureTokenizer(nn.Module):
    def __init__(self, n_features: int, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, dim))
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.unsqueeze(-1) * self.weight + self.bias
        cls = self.cls.expand(x.shape[0], -1, -1)
        return torch.cat([cls, tokens], dim=1)


class _EncoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, need_weights: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = self.norm1(x)
        attended, weights = self.attn(
            h, h, h, need_weights=need_weights, average_attn_weights=False
        )
        x = x + self.drop(attended)
        x = x + self.ff(self.norm2(x))
        return x, weights


class FTTransformerEncoder(nn.Module):
    def __init__(self, cfg: FTConfig) -> None:
        super().__init__()
        self.tokenizer = FeatureTokenizer(cfg.n_features, cfg.dim)
        self.layers = nn.ModuleList(
            [_EncoderLayer(cfg.dim, cfg.heads, cfg.ff_mult, cfg.dropout) for _ in range(cfg.layers)]
        )
        self.norm = nn.LayerNorm(cfg.dim)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        tokens = self.tokenizer(x)
        attention: torch.Tensor | None = None
        last = len(self.layers) - 1
        for idx, layer in enumerate(self.layers):
            tokens, weights = layer(tokens, need_weights=return_attention and idx == last)
            if weights is not None:
                attention = weights
        tokens = self.norm(tokens)
        return tokens[:, 0], attention


class FTTransformerClassifier(nn.Module):
    def __init__(self, cfg: FTConfig) -> None:
        super().__init__()
        self.encoder = FTTransformerEncoder(cfg)
        self.head = nn.Linear(cfg.dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls, _ = self.encoder(x)
        return self.head(cls).squeeze(-1)
