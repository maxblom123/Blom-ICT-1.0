import math
from dataclasses import dataclass

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from blom.data import Shard, eval_batches, train_batch
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


def train_step(
    model: GPT,
    optimizer: AdamW,
    shard: Shard,
    config: TrainConfig,
    *,
    step: int,
    device: torch.device,
    amp: bool = False,
) -> StepStats:
    lr = config.schedule.lr_at(step)
    set_lr(optimizer, lr)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = torch.zeros((), device=device)
    for micro in range(config.grad_accum):
        inputs, targets = train_batch(
            shard,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            seed=config.seed,
            step=step * config.grad_accum + micro,
        )
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast(device, amp):
            logits = model(inputs)
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
) -> float:
    was_training = model.training
    model.eval()
    total = 0.0
    count = 0
    try:
        with torch.no_grad():
            batches = eval_batches(shard, batch_size=config.batch_size, seq_len=config.seq_len)
            for index, (inputs, targets) in enumerate(batches):
                if index >= config.eval_batches:
                    break
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with autocast(device, amp):
                    logits = model(inputs)
                total += float(next_token_loss(logits, targets)) * targets.numel()
                count += targets.numel()
    finally:
        model.train(was_training)
    return total / count
