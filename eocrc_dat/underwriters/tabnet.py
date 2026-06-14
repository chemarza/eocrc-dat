from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from eocrc_dat.underwriters.sparsemax import sparsemax


@dataclass(frozen=True)
class TabNetConfig:
    n_features: int
    n_steps: int = 5
    gamma: float = 1.3
    feature_dim: int = 64
    n_glu: int = 2


@dataclass
class TabNetOutput:
    representation: torch.Tensor
    importance: torch.Tensor
    masks: torch.Tensor


class _GLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, 2 * out_dim)
        self.bn = nn.BatchNorm1d(2 * out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.glu(self.bn(self.fc(x)), dim=-1)


class _FeatureTransformer(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_glu: int) -> None:
        super().__init__()
        blocks: list[_GLU] = []
        width = in_dim
        for _ in range(n_glu):
            blocks.append(_GLU(width, hidden))
            width = hidden
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for idx, block in enumerate(self.blocks):
            transformed = block(out)
            out = transformed if idx == 0 else (transformed + out) * math.sqrt(0.5)
        return out


class _AttentiveTransformer(nn.Module):
    def __init__(self, hidden: int, n_features: int) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden, n_features)
        self.bn = nn.BatchNorm1d(n_features)

    def forward(self, a: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        logits = self.bn(self.fc(a)) * prior
        return sparsemax(logits, dim=-1)


class TabNetEncoder(nn.Module):
    def __init__(self, cfg: TabNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_bn = nn.BatchNorm1d(cfg.n_features)
        self.initial = _FeatureTransformer(cfg.n_features, cfg.feature_dim, cfg.n_glu)
        self.steps = nn.ModuleList(
            [
                _FeatureTransformer(cfg.n_features, cfg.feature_dim, cfg.n_glu)
                for _ in range(cfg.n_steps)
            ]
        )
        self.attentive = nn.ModuleList(
            [_AttentiveTransformer(cfg.feature_dim, cfg.n_features) for _ in range(cfg.n_steps)]
        )

    def forward(self, x: torch.Tensor) -> TabNetOutput:
        normalized = self.input_bn(x)
        a = self.initial(normalized)
        prior = torch.ones_like(x)
        importance = torch.zeros_like(x)
        representation = torch.zeros(x.shape[0], self.cfg.feature_dim, device=x.device)
        masks: list[torch.Tensor] = []
        for step, attentive in zip(self.steps, self.attentive, strict=True):
            mask = attentive(a, prior)
            masks.append(mask)
            prior = prior * (self.cfg.gamma - mask)
            decision = F.relu(step(mask * normalized))
            representation = representation + decision
            importance = importance + mask * decision.sum(dim=-1, keepdim=True)
            a = decision
        return TabNetOutput(representation, importance, torch.stack(masks, dim=0))


class TabNetClassifier(nn.Module):
    def __init__(self, cfg: TabNetConfig) -> None:
        super().__init__()
        self.encoder = TabNetEncoder(cfg)
        self.head = nn.Linear(cfg.feature_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder(x)
        return self.head(out.representation).squeeze(-1)
