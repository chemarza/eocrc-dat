from __future__ import annotations

import math
from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def cosine_warmup(optimizer: Optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    floor = max(total_steps, 1)

    def factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(floor - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler: Any = LambdaLR(optimizer, lr_lambda=factor)
    return scheduler
