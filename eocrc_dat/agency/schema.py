from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    method: str = "dat"
    k_steps: int = 5
    gamma: float = 1.3
    tabnet_dim: int = 64
    n_glu: int = 2
    q_select: int = 32
    ft_dim: int = 192
    ft_heads: int = 8
    ft_layers: int = 3
    ff_mult: int = 2
    dropout: float = 0.1
    mlp_hidden: tuple[int, ...] = (256, 128)
    saint_dim: int = 96
    saint_heads: int = 8
    saint_layers: int = 2


@dataclass(frozen=True)
class DataConfig:
    source: str = "synthetic"
    match: bool = True
    winsor_p: float = 0.01
    scale: float = 1.0
    seed: int = 7
    drop_domains: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrainConfig:
    max_epochs: int = 200
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    patience: int = 20
    grad_accum: int = 1
    world_size: int = 1
    amp: bool = False
    lam: float = 1e-5
    max_steps: int | None = None
    device: str = "auto"

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum * self.world_size


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int = 7
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    thresholds: tuple[float, ...] = (0.01, 0.02, 0.05)
