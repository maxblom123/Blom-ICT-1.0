from dataclasses import dataclass

from torch import nn
from torch.optim import AdamW

from blom.model import GPT


class ScheduleError(ValueError):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class Schedule:
    peak_lr: float
    min_lr: float
    warmup_steps: int
    decay_steps: int
    total_steps: int

    def __post_init__(self) -> None:
        if self.warmup_steps <= 0 or self.decay_steps <= 0:
            raise ScheduleError("warmup_steps and decay_steps must be positive")
        if self.warmup_steps + self.decay_steps > self.total_steps:
            raise ScheduleError("warmup_steps + decay_steps must not exceed total_steps")
        if not 0 <= self.min_lr <= self.peak_lr or self.peak_lr <= 0:
            raise ScheduleError("need 0 <= min_lr <= peak_lr and peak_lr > 0")

    def lr_at(self, step: int) -> float:
        if not 0 <= step < self.total_steps:
            raise ScheduleError(f"step {step} is outside 0..{self.total_steps - 1}")
        if step < self.warmup_steps:
            return self.peak_lr * ((step + 1) / self.warmup_steps)
        decay_start = self.total_steps - self.decay_steps
        if step < decay_start:
            return self.peak_lr
        progress = (step - decay_start + 1) / self.decay_steps
        return self.min_lr + (self.peak_lr - self.min_lr) * (1 - progress)


def build_optimizer(
    model: GPT,
    *,
    lr: float,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    fused: bool = False,
) -> AdamW:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or parameter is model.tok.weight:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return AdamW(groups, lr=lr, betas=betas, eps=eps, fused=fused)


def set_lr(optimizer: AdamW, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr
