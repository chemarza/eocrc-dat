from __future__ import annotations

import os
from typing import Any

from torch import nn

from eocrc_dat.support.io import load_checkpoint, save_atomic


def save_checkpoint(
    model: nn.Module,
    seed: int,
    metadata: dict[str, Any],
    path: str | os.PathLike[str],
) -> None:
    state = {
        "model_state": model.state_dict(),
        "seed": seed,
        "metadata": metadata,
    }
    save_atomic(state, path)


def restore(model: nn.Module, path: str | os.PathLike[str]) -> dict[str, Any]:
    state = load_checkpoint(path)
    model.load_state_dict(state["model_state"])
    return state
