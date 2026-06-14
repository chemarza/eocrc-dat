from __future__ import annotations

import torch


def pick_device(prefer: str = "auto") -> torch.device:
    if prefer not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"unknown device preference: {prefer}")
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        return torch.device("cuda")
    if prefer == "mps":
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
