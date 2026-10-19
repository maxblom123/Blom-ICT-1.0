import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path

from blom.optim import Schedule, ScheduleError

INT_FIELDS = (
    "seed",
    "batch_size",
    "grad_accum",
    "seq_len",
    "warmup_steps",
    "decay_steps",
    "total_steps",
    "eval_every",
    "eval_batches",
    "checkpoint_every",
    "log_every",
)
FLOAT_FIELDS = ("peak_lr", "min_lr", "weight_decay", "max_grad_norm")
POSITIVE_FIELDS = (
    "batch_size",
    "grad_accum",
    "seq_len",
    "eval_every",
    "eval_batches",
    "checkpoint_every",
    "log_every",
    "max_grad_norm",
)
NON_NEGATIVE_FIELDS = ("seed", "weight_decay")


class TrainConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainConfig:
    batch_size: int
    seq_len: int
    peak_lr: float
    min_lr: float
    warmup_steps: int
    decay_steps: int
    total_steps: int
    grad_accum: int = 1
    seed: int = 0
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    eval_every: int = 500
    eval_batches: int = 50
    checkpoint_every: int = 500
    log_every: int = 10

    def __post_init__(self) -> None:
        for name in INT_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TrainConfigError(f"{name} must be an integer, got {value!r}")
        for name in FLOAT_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TrainConfigError(f"{name} must be a number, got {value!r}")
            if not math.isfinite(value):
                raise TrainConfigError(f"{name} must be finite, got {value!r}")
        for name in POSITIVE_FIELDS:
            if getattr(self, name) <= 0:
                raise TrainConfigError(f"{name} must be positive")
        for name in NON_NEGATIVE_FIELDS:
            if getattr(self, name) < 0:
                raise TrainConfigError(f"{name} must not be negative")
        try:
            self.schedule  # noqa: B018
        except ScheduleError as error:
            raise TrainConfigError(str(error)) from error

    @property
    def schedule(self) -> Schedule:
        return Schedule(
            peak_lr=self.peak_lr,
            min_lr=self.min_lr,
            warmup_steps=self.warmup_steps,
            decay_steps=self.decay_steps,
            total_steps=self.total_steps,
        )

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.grad_accum * self.seq_len

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "TrainConfig":
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise TrainConfigError(f"unknown train config keys: {', '.join(unknown)}")
        try:
            return cls(**data)  # pyright: ignore[reportArgumentType]
        except TypeError as error:
            raise TrainConfigError(str(error)) from error

    @classmethod
    def load(cls, path: Path) -> "TrainConfig":
        try:
            with path.open("rb") as handle:
                document = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError) as error:
            raise TrainConfigError(f"cannot read {path}: {error}") from error
        table = document.get("train")
        if not isinstance(table, dict):
            raise TrainConfigError(f"{path} has no [train] table")
        return cls.from_dict(table)
