import tomllib
import typing
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path


class ConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelConfig:
    vocab_size: int
    d_model: int
    n_layer: int
    n_head: int
    n_kv_head: int
    d_ff: int
    max_seq: int
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-6
    qk_norm: bool = True
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        hints = typing.get_type_hints(type(self))
        for field in fields(self):
            self._check_type(field.name, hints[field.name])
        for name in ("vocab_size", "d_model", "n_layer", "n_head", "n_kv_head", "d_ff", "max_seq"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} must be positive, got {getattr(self, name)}")
        if self.rope_theta <= 0 or self.norm_eps <= 0:
            raise ConfigError("rope_theta and norm_eps must be positive")
        if self.d_model % self.n_head != 0:
            raise ConfigError(f"d_model {self.d_model} must be divisible by n_head {self.n_head}")
        if self.head_dim % 2 != 0:
            raise ConfigError(f"head_dim {self.head_dim} must be even for rotary embeddings")
        if self.n_head % self.n_kv_head != 0:
            raise ConfigError(
                f"n_head {self.n_head} must be divisible by n_kv_head {self.n_kv_head}"
            )

    def _check_type(self, name: str, expected: type) -> None:
        value = getattr(self, name)
        if expected is float:
            valid = isinstance(value, int | float) and not isinstance(value, bool)
        elif expected is int:
            valid = isinstance(value, int) and not isinstance(value, bool)
        else:
            valid = isinstance(value, expected)
        if not valid:
            raise ConfigError(f"{name} must be {expected.__name__}, got {value!r}")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    @property
    def kv_dim(self) -> int:
        return self.n_kv_head * self.head_dim

    @property
    def param_count(self) -> int:
        d = self.d_model
        attention = d * d + 2 * d * self.kv_dim + d * d
        if self.qk_norm:
            attention += 2 * self.head_dim
        block = attention + 2 * d + 3 * d * self.d_ff
        total = self.vocab_size * d + self.n_layer * block + d
        if not self.tie_embeddings:
            total += self.vocab_size * d
        return total

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ModelConfig":
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ConfigError(f"unknown model config keys: {', '.join(unknown)}")
        try:
            return cls(**data)  # pyright: ignore[reportArgumentType]
        except TypeError as error:
            raise ConfigError(str(error)) from error

    @classmethod
    def load(cls, path: Path) -> "ModelConfig":
        with path.open("rb") as handle:
            document = tomllib.load(handle)
        table = document.get("model")
        if not isinstance(table, dict):
            raise ConfigError(f"{path} has no [model] table")
        return cls.from_dict(table)
