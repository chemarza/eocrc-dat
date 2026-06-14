from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from eocrc_dat.ledger.records import CohortArrays
from eocrc_dat.objective.losses import bce_with_l2
from eocrc_dat.reserving.schedule import cosine_warmup
from eocrc_dat.support.logging import get_logger

_LOG = get_logger("eocrc_dat.reserving")


@dataclass(frozen=True)
class TrainSettings:
    max_epochs: int = 200
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    patience: int = 20
    grad_accum: int = 1
    amp: bool = False
    seed: int = 7
    max_steps: int | None = None


@dataclass
class TrainResult:
    best_auroc: float
    epochs_run: int
    step_losses: list[float] = field(default_factory=list)
    best_state: dict[str, torch.Tensor] = field(default_factory=dict)


def _loader(
    arrays: CohortArrays, batch_size: int, shuffle: bool
) -> DataLoader[tuple[torch.Tensor, ...]]:
    tensors = TensorDataset(
        torch.from_numpy(arrays.x), torch.from_numpy(arrays.y.astype(np.float32))
    )
    return DataLoader(tensors, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


def predict_proba(
    model: nn.Module, arrays: CohortArrays, device: torch.device, batch_size: int = 512
) -> npt.NDArray[np.float64]:
    model.eval()
    outputs: list[npt.NDArray[np.float64]] = []
    with torch.no_grad():
        for start in range(0, arrays.n, batch_size):
            chunk = torch.from_numpy(arrays.x[start : start + batch_size]).to(device)
            logits = model(chunk)
            outputs.append(torch.sigmoid(logits).cpu().numpy().astype(np.float64))
    return np.concatenate(outputs)


def train_model(
    model: nn.Module,
    train: CohortArrays,
    val: CohortArrays,
    settings: TrainSettings,
    device: torch.device,
    lam: float = 1e-5,
) -> TrainResult:
    model.to(device)
    loader = _loader(train, settings.batch_size, shuffle=True)
    steps_per_epoch = max(len(loader), 1)
    total_steps = steps_per_epoch * settings.max_epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=0.0)
    scheduler = cosine_warmup(optimizer, int(settings.warmup_ratio * total_steps), total_steps)

    best_auroc = -1.0
    best_state: dict[str, torch.Tensor] = {}
    stale = 0
    step_losses: list[float] = []
    step_count = 0
    epoch = 0
    stop = False
    while epoch < settings.max_epochs and not stop:
        model.train()
        optimizer.zero_grad()
        for batch_idx, (features, targets) in enumerate(loader):
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            loss = bce_with_l2(logits, targets, model.parameters(), lam) / settings.grad_accum
            torch.autograd.backward(loss)
            if (batch_idx + 1) % settings.grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            step_losses.append(float(loss.item()) * settings.grad_accum)
            step_count += 1
            if settings.max_steps is not None and step_count >= settings.max_steps:
                stop = True
                break
        probabilities = predict_proba(model, val, device)
        auroc = float(roc_auc_score(val.y, probabilities)) if val.y.min() != val.y.max() else 0.5
        if auroc > best_auroc:
            best_auroc = auroc
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        epoch += 1
        if stale >= settings.patience:
            break
    _LOG.info("training finished: epochs=%d best_auroc=%.4f", epoch, best_auroc)
    return TrainResult(
        best_auroc=best_auroc, epochs_run=epoch, step_losses=step_losses, best_state=best_state
    )
