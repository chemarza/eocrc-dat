from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F


def bce_with_l2(
    logits: torch.Tensor, target: torch.Tensor, params: Iterable[torch.Tensor], lam: float
) -> torch.Tensor:
    data_term = F.binary_cross_entropy_with_logits(logits, target.to(logits.dtype))
    penalty = logits.new_zeros(())
    for tensor in params:
        penalty = penalty + tensor.pow(2).sum()
    return data_term + lam * penalty
