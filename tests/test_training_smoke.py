from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from eocrc_dat.agency.commands import run_train
from eocrc_dat.agency.config import load_experiment
from eocrc_dat.underwriters.dat import DATConfig, DualAttentionTransformer


def test_smoke_experiment_runs(configs_dir: Path) -> None:
    cfg = load_experiment("_smoke", configs_dir)
    summary = run_train(cfg)
    assert summary["method"] == "dat"
    assert 0.0 <= summary["internal_test_auroc"] <= 1.0
    assert summary["epochs_run"] >= 1


def test_two_step_loss_decreases() -> None:
    torch.manual_seed(0)
    model = DualAttentionTransformer(
        DATConfig(
            n_features=40, k_steps=2, tabnet_dim=16, q_select=20, ft_dim=32, ft_heads=4, ft_layers=2
        )
    )
    x = torch.randn(64, 40)
    y = (torch.rand(64) > 0.5).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    model.train()
    losses: list[float] = []
    for _ in range(20):
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5
