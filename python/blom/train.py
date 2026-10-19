import math
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from blom.data import Shard, eval_batches, train_batch
from blom.memory import SketchReader, eval_batches_with_memory, train_batch_with_memory
from blom.model import GPT, next_token_loss
from blom.optim import set_lr
from blom.train_config import TrainConfig


class TrainError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class StepStats:
    step: int
    loss: float
    grad_norm: float
    lr: float
    tokens: int


def autocast(device: torch.device, enabled: bool) -> torch.autocast:
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


def _train_batch(
    shard: Shard, config: TrainConfig, memory: SketchReader | None, step: int
) -> tuple[Tensor, Tensor, Tensor | None]:
    if memory is None:
        inputs, targets = train_batch(
            shard,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            seed=config.seed,
            step=step,
        )
        return inputs, targets, None
    return train_batch_with_memory(
        shard,
        memory,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        seed=config.seed,
        step=step,
    )


def _eval_batches(
    shard: Shard, config: TrainConfig, memory: SketchReader | None
) -> Iterator[tuple[Tensor, Tensor, Tensor | None]]:
    if memory is None:
        for inputs, targets in eval_batches(
            shard, batch_size=config.batch_size, seq_len=config.seq_len
        ):
            yield inputs, targets, None
    else:
        yield from eval_batches_with_memory(
            shard, memory, batch_size=config.batch_size, seq_len=config.seq_len
        )


def train_step(
    model: GPT,
    optimizer: AdamW,
    shard: Shard,
    config: TrainConfig,
    *,
    step: int,
    device: torch.device,
    amp: bool = False,
    memory: SketchReader | None = None,
) -> StepStats:
    lr = config.schedule.lr_at(step)
    set_lr(optimizer, lr)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = torch.zeros((), device=device)
    for micro in range(config.grad_accum):
        inputs, targets, mem = _train_batch(shard, config, memory, step * config.grad_accum + micro)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mem = None if mem is None else mem.to(device, non_blocking=True)
        with autocast(device, amp):
            logits = model(inputs, mem=mem)
        loss = next_token_loss(logits, targets) / config.grad_accum
        loss.backward()
        total += loss.detach()
    loss_value = float(total)
    if not math.isfinite(loss_value):
        raise TrainError(f"loss is {loss_value} at step {step}")
    grad_norm = float(clip_grad_norm_(model.parameters(), config.max_grad_norm))
    if not math.isfinite(grad_norm):
        raise TrainError(f"gradient norm is {grad_norm} at step {step}")
    optimizer.step()
    return StepStats(
        step=step,
        loss=loss_value,
        grad_norm=grad_norm,
        lr=lr,
        tokens=config.tokens_per_step,
    )


def evaluate(
    model: GPT,
    shard: Shard,
    config: TrainConfig,
    *,
    device: torch.device,
    amp: bool = False,
    memory: SketchReader | None = None,
) -> float:
    was_training = model.training
    model.eval()
    total = 0.0
    count = 0
    try:
        with torch.no_grad():
            for index, (inputs, targets, mem) in enumerate(_eval_batches(shard, config, memory)):
                if index >= config.eval_batches:
                    break
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                mem = None if mem is None else mem.to(device, non_blocking=True)
                with autocast(device, amp):
                    logits = model(inputs, mem=mem)
                total += float(next_token_loss(logits, targets)) * targets.numel()
                count += targets.numel()
    finally:
        model.train(was_training)
    return total / count
