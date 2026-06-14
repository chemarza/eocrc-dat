from __future__ import annotations

import torch
import torch.nn.functional as F

from eocrc_dat.underwriters.dat import DATConfig, DualAttentionTransformer


def _model() -> DualAttentionTransformer:
    return DualAttentionTransformer(
        DATConfig(
            n_features=40, k_steps=2, tabnet_dim=16, q_select=20, ft_dim=32, ft_heads=4, ft_layers=2
        )
    )


def test_overfits_single_batch() -> None:
    torch.manual_seed(0)
    model = _model()
    x = torch.randn(48, 40)
    y = (torch.rand(48) > 0.5).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    model.train()
    first = None
    last = 0.0
    for step in range(80):
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(x), y)
        loss.backward()
        optimizer.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    assert first is not None
    assert last < 0.5 * first


def test_gradient_reaches_tabnet() -> None:
    torch.manual_seed(0)
    model = _model()
    x = torch.randn(32, 40)
    y = (torch.rand(32) > 0.5).float()
    loss = F.binary_cross_entropy_with_logits(model(x), y)
    loss.backward()
    grads = [
        param.grad is not None and float(param.grad.abs().sum()) > 0.0
        for name, param in model.named_parameters()
        if "tabnet" in name
    ]
    assert grads and all(grads)
