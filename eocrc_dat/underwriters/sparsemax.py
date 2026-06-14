from __future__ import annotations

import torch


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    z = logits.transpose(dim, -1)
    shape = z.shape
    flat = z.reshape(-1, shape[-1])
    sorted_z, _ = torch.sort(flat, descending=True, dim=-1)
    cumulative = sorted_z.cumsum(dim=-1) - 1.0
    span = torch.arange(1, flat.shape[-1] + 1, device=flat.device, dtype=flat.dtype)
    support = (sorted_z - cumulative / span) > 0
    k = support.to(flat.dtype).sum(dim=-1, keepdim=True)
    index = k.long().clamp(min=1) - 1
    tau = cumulative.gather(-1, index) / k.clamp(min=1.0)
    projected = torch.clamp(flat - tau, min=0.0)
    return projected.reshape(shape).transpose(dim, -1)
