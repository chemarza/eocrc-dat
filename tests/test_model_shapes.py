from __future__ import annotations

import torch

from eocrc_dat.underwriters.dat import DATConfig, DualAttentionTransformer
from eocrc_dat.underwriters.ft_transformer import FTConfig, FTTransformerClassifier
from eocrc_dat.underwriters.mlp import MLPClassifier, MLPConfig
from eocrc_dat.underwriters.saint import SaintClassifier, SaintConfig
from eocrc_dat.underwriters.tabnet import TabNetClassifier, TabNetConfig

FEATURES = 91
BATCH = 12


def _x() -> torch.Tensor:
    return torch.randn(BATCH, FEATURES)


def test_dat_shape() -> None:
    model = DualAttentionTransformer(
        DATConfig(n_features=FEATURES, k_steps=2, tabnet_dim=16, ft_dim=32, ft_heads=4, ft_layers=2)
    )
    assert model(_x()).shape == (BATCH,)


def test_ft_tabnet_saint_mlp_shapes() -> None:
    builders = [
        FTTransformerClassifier(FTConfig(n_features=FEATURES, dim=32, heads=4, layers=2)),
        TabNetClassifier(TabNetConfig(n_features=FEATURES, n_steps=2, feature_dim=16)),
        SaintClassifier(SaintConfig(n_features=FEATURES, dim=32, heads=4, layers=2)),
        MLPClassifier(MLPConfig(n_features=FEATURES, hidden=(32, 16))),
    ]
    for model in builders:
        assert model(_x()).shape == (BATCH,)


def test_dat_trace_dimensions() -> None:
    model = DualAttentionTransformer(
        DATConfig(
            n_features=FEATURES,
            k_steps=2,
            tabnet_dim=16,
            q_select=20,
            ft_dim=32,
            ft_heads=4,
            ft_layers=2,
        )
    )
    trace = model.trace(_x())
    assert trace.attention.shape[-1] == FEATURES + 1
    assert trace.selected.shape == (20,)
