from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch


def save_atomic(state: dict[str, Any], path: str | os.PathLike[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    os.close(fd)
    try:
        torch.save(state, tmp)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_checkpoint(path: str | os.PathLike[str]) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise ValueError(f"checkpoint at {path} is not a state mapping")
    return obj


def write_json(obj: dict[str, Any], path: str | os.PathLike[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)
    os.replace(tmp, target)
