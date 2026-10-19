import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.optim import AdamW

from blom.config import ConfigError, ModelConfig
from blom.model import GPT

FORMAT = 1
REQUIRED_KEYS = frozenset({"format", "step", "config", "model", "optimizer"})
ALLOWED_KEYS = REQUIRED_KEYS | {"memory"}


class CheckpointError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Checkpoint:
    step: int
    config: ModelConfig
    model: dict[str, Tensor]
    optimizer: dict[str, Any]
    memory: dict[str, Any] | None = None


def save_checkpoint(
    path: Path,
    *,
    model: GPT,
    optimizer: AdamW,
    step: int,
    memory: Mapping[str, object] | None = None,
) -> None:
    if step < 0:
        raise CheckpointError(f"step must be non-negative, got {step}")
    payload: dict[str, object] = {
        "format": FORMAT,
        "step": step,
        "config": asdict(model.config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if memory is not None:
        payload["memory"] = dict(memory)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_checkpoint(path: Path, *, map_location: str | torch.device = "cpu") -> Checkpoint:
    if not path.is_file():
        raise CheckpointError(f"{path} is not a file")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as error:
        raise CheckpointError(f"cannot read {path}: {error}") from error
    if not isinstance(payload, dict) or not REQUIRED_KEYS <= set(payload) <= ALLOWED_KEYS:
        raise CheckpointError(f"{path} is not a checkpoint")
    if payload["format"] != FORMAT:
        raise CheckpointError(f"{path} has format {payload['format']}, expected {FORMAT}")
    step = payload["step"]
    if not isinstance(step, int) or step < 0:
        raise CheckpointError(f"{path} has an invalid step {step!r}")
    try:
        config = ModelConfig.from_dict(payload["config"])
    except ConfigError as error:
        raise CheckpointError(f"{path} has an invalid config: {error}") from error
    memory = payload.get("memory")
    if memory is not None and not isinstance(memory, dict):
        raise CheckpointError(f"{path} has an invalid memory description")
    return Checkpoint(
        step=step,
        config=config,
        model=payload["model"],
        optimizer=payload["optimizer"],
        memory=memory,
    )


def restore(checkpoint: Checkpoint, *, model: GPT, optimizer: AdamW | None = None) -> int:
    if checkpoint.config != model.config:
        raise CheckpointError("checkpoint config does not match the model config")
    try:
        model.load_state_dict(checkpoint.model, strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint.optimizer)
    except (RuntimeError, ValueError, KeyError) as error:
        raise CheckpointError(f"checkpoint does not fit the model: {error}") from error
    return checkpoint.step
