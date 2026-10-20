import argparse
import contextlib
import json
import math
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from blom.checkpoint import CheckpointError, load_checkpoint, restore, save_checkpoint
from blom.data import BYTES_PER_TOKEN, TOKEN_DTYPE, DataError, Dataset, Shard
from blom.model import GPT, IGNORE_INDEX
from blom.optim import build_optimizer, set_lr
from blom.run import (
    CHECKPOINT_NAME,
    METRICS_NAME,
    check_run_config,
    peak_memory_mb,
    resolve_device,
)
from blom.train import TrainError, autocast
from blom.train_config import TrainConfig, TrainConfigError

MASK_DTYPE = np.dtype("u1")
STARTS_SUFFIX = ".starts.npy"
SCAN_CHUNK = 1 << 24
CANDIDATES_PER_SLOT = 8

Log = Callable[[dict[str, object]], None]


class ChatShard:
    """Tokens plus a one-byte-per-token mask: 1 where the model should learn."""

    def __init__(self, tokens: Shard, mask_path: Path, end_id: int) -> None:
        try:
            size = mask_path.stat().st_size
        except OSError as error:
            raise DataError(f"cannot read mask {mask_path}: {error}") from error
        if size != len(tokens):
            raise DataError(f"{mask_path} has {size} bytes, expected {len(tokens)}")
        self.tokens = tokens
        self.mask_path = mask_path
        self.end_id = end_id
        self._starts: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.tokens)

    def mask_window(self, start: int, length: int) -> np.ndarray:
        if start < 0 or length <= 0 or start + length > len(self):
            raise DataError(f"mask window {start}+{length} is outside {len(self)} tokens")
        return np.fromfile(self.mask_path, dtype=MASK_DTYPE, count=length, offset=start)

    def _scan_starts(self) -> np.ndarray:
        found = [np.zeros(1, dtype=np.int64)]
        total = len(self)
        for first in range(0, total, SCAN_CHUNK):
            count = min(SCAN_CHUNK, total - first)
            chunk = np.fromfile(
                self.tokens.path, dtype=TOKEN_DTYPE, count=count, offset=first * BYTES_PER_TOKEN
            )
            found.append(np.flatnonzero(chunk == self.end_id).astype(np.int64) + first + 1)
        starts = np.concatenate(found)
        return np.unique(starts[starts < total])

    def _cache_is_valid(self, cache: Path, starts: np.ndarray) -> bool:
        if cache.stat().st_mtime < self.tokens.path.stat().st_mtime:
            return False
        if starts.ndim != 1 or starts.size == 0 or int(starts[0]) != 0:
            return False
        if int(starts[-1]) >= len(self) or bool(np.any(np.diff(starts) <= 0)):
            return False
        last = int(starts[-1])
        return last == 0 or int(self.tokens.window(last - 1, 1)[0]) == self.end_id

    @property
    def starts(self) -> np.ndarray:
        if self._starts is not None:
            return self._starts
        cache = self.tokens.path.with_name(self.tokens.path.stem + STARTS_SUFFIX)
        found: np.ndarray | None = None
        if cache.is_file():
            try:
                loaded = np.load(cache, allow_pickle=False)
            except (OSError, ValueError):
                loaded = None
            if loaded is not None and self._cache_is_valid(cache, loaded):
                found = loaded.astype(np.int64)
        if found is None:
            found = self._scan_starts()
            with contextlib.suppress(OSError):
                np.save(cache, found)
        self._starts = found
        return found

    def usable_starts(self, seq_len: int) -> np.ndarray:
        starts = self.starts
        usable = starts[starts + seq_len + 1 <= len(self)]
        if usable.size == 0:
            raise DataError(f"no conversation leaves room for {seq_len + 1} tokens")
        return usable

    def window(self, start: int, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
        tokens = self.tokens.window(start, seq_len + 1)
        mask = self.mask_window(start, seq_len + 1)
        inputs = tokens[:-1]
        targets = np.where(mask[1:] == 1, tokens[1:], IGNORE_INDEX)
        return inputs, targets


def _stack(rows: list[tuple[np.ndarray, np.ndarray]]) -> tuple[Tensor, Tensor]:
    inputs = torch.from_numpy(np.stack([row[0] for row in rows]))
    targets = torch.from_numpy(np.stack([row[1] for row in rows]))
    return inputs, targets


def train_batch(
    shard: ChatShard, *, batch_size: int, seq_len: int, seed: int, step: int
) -> tuple[Tensor, Tensor]:
    usable = shard.usable_starts(seq_len)
    rng = np.random.default_rng((seed, step))
    picks = rng.integers(0, usable.size, size=batch_size * CANDIDATES_PER_SLOT)
    rows: list[tuple[np.ndarray, np.ndarray]] = []
    for pick in picks:
        inputs, targets = shard.window(int(usable[pick]), seq_len)
        if bool(np.any(targets != IGNORE_INDEX)):
            rows.append((inputs, targets))
            if len(rows) == batch_size:
                return _stack(rows)
    raise DataError("could not find enough windows with assistant text to learn from")


def eval_batches(
    shard: ChatShard, *, batch_size: int, seq_len: int, windows: int
) -> Iterator[tuple[Tensor, Tensor]]:
    usable = shard.usable_starts(seq_len)
    count = min(usable.size, windows)
    chosen = np.unique(np.linspace(0, usable.size - 1, num=count).astype(np.int64))
    for first in range(0, chosen.size, batch_size):
        picks = chosen[first : first + batch_size]
        yield _stack([shard.window(int(usable[pick]), seq_len) for pick in picks])


def trained_count(targets: Tensor) -> int:
    return int((targets != IGNORE_INDEX).sum())


def loss_sum(logits: Tensor, targets: Tensor) -> Tensor:
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )


@dataclass(frozen=True, slots=True)
class SftStats:
    step: int
    loss: float
    grad_norm: float
    lr: float
    tokens: int
    trained_tokens: int


def sft_step(
    model: GPT,
    optimizer: AdamW,
    shard: ChatShard,
    config: TrainConfig,
    *,
    step: int,
    device: torch.device,
    amp: bool = False,
) -> SftStats:
    lr = config.schedule.lr_at(step)
    set_lr(optimizer, lr)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    batches = [
        train_batch(
            shard,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            seed=config.seed,
            step=step * config.grad_accum + micro,
        )
        for micro in range(config.grad_accum)
    ]
    trained = sum(trained_count(targets) for _, targets in batches)
    total = torch.zeros((), device=device)
    for inputs, targets in batches:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast(device, amp):
            logits = model(inputs)
        loss = loss_sum(logits, targets) / trained
        loss.backward()
        total += loss.detach()
    loss_value = float(total)
    if not math.isfinite(loss_value):
        raise TrainError(f"loss is {loss_value} at step {step}")
    grad_norm = float(clip_grad_norm_(model.parameters(), config.max_grad_norm))
    if not math.isfinite(grad_norm):
        raise TrainError(f"gradient norm is {grad_norm} at step {step}")
    optimizer.step()
    return SftStats(
        step=step,
        loss=loss_value,
        grad_norm=grad_norm,
        lr=lr,
        tokens=config.tokens_per_step,
        trained_tokens=trained,
    )


def sft_evaluate(
    model: GPT,
    shard: ChatShard,
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
            for inputs, targets in eval_batches(
                shard,
                batch_size=config.batch_size,
                seq_len=config.seq_len,
                windows=config.eval_batches * config.batch_size,
            ):
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with autocast(device, amp):
                    logits = model(inputs)
                total += float(loss_sum(logits, targets))
                count += trained_count(targets)
    finally:
        model.train(was_training)
    if count == 0:
        raise DataError("the validation data has no assistant text to score")
    return total / count


@dataclass(frozen=True, slots=True)
class SftResult:
    step: int
    train_loss: float | None
    val_loss: float | None


def open_chat_data(data_dir: Path, *, vocab_size: int) -> tuple[ChatShard, ChatShard]:
    dataset = Dataset.open(data_dir, expected_vocab=vocab_size)
    try:
        meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataError(f"cannot read meta.json in {data_dir}: {error}") from error
    if meta.get("kind") != "chat" or meta.get("mask_format") != "u8":
        raise DataError(f"{data_dir} was not made by dataprep tokenize-chat")
    end_id = dataset.end_of_text_id
    return (
        ChatShard(dataset.train, data_dir / "train.mask", end_id),
        ChatShard(dataset.val, data_dir / "val.mask", end_id),
    )


def sft_run(
    train_config: TrainConfig,
    *,
    init: Path | None,
    data_dir: Path,
    out_dir: Path,
    device: torch.device,
    amp: bool,
    log: Log,
    until: int | None = None,
) -> SftResult:
    end = train_config.total_steps if until is None else until
    if not 0 < end <= train_config.total_steps:
        raise TrainError(f"until must be in 1..{train_config.total_steps}, got {end}")
    checkpoint_path = out_dir / CHECKPOINT_NAME
    resuming = checkpoint_path.exists()
    source = checkpoint_path if resuming else init
    if source is None:
        raise TrainError("--init is needed to start a new fine-tuning run")
    checkpoint = load_checkpoint(source, map_location=device)
    model_config = checkpoint.config
    if model_config.memory_dim:
        raise TrainError("fine-tuning a model with sketch memory is not supported")
    if train_config.seq_len > model_config.max_seq:
        raise TrainError(
            f"seq_len {train_config.seq_len} exceeds the model's max_seq {model_config.max_seq}"
        )
    train_shard, val_shard = open_chat_data(data_dir, vocab_size=model_config.vocab_size)
    out_dir.mkdir(parents=True, exist_ok=True)
    check_run_config(out_dir, model_config, train_config)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(train_config.seed)
    model = GPT(model_config).to(device)
    optimizer = build_optimizer(
        model,
        lr=train_config.peak_lr,
        weight_decay=train_config.weight_decay,
        fused=device.type == "cuda",
    )
    start = 0
    if resuming:
        start = restore(checkpoint, model=model, optimizer=optimizer)
    else:
        restore(checkpoint, model=model)
    log(
        {
            "event": "start",
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "amp": amp,
            "parameters": model_config.param_count,
            "init": None if resuming or init is None else str(init),
            "tokens_per_step": train_config.tokens_per_step,
            "total_steps": train_config.total_steps,
            "until": end,
            "train_conversations": int(train_shard.starts.size),
        }
    )
    if resuming:
        log({"event": "resume", "step": start})
    if start >= end:
        log({"event": "nothing_to_do", "step": start, "until": end})
        return SftResult(step=start, train_loss=None, val_loss=None)
    if start == 0:
        first = sft_evaluate(model, val_shard, train_config, device=device, amp=amp)
        log({"event": "eval", "step": 0, "val_loss": round(first, 4)})

    train_loss: float | None = None
    val_loss: float | None = None
    window_tokens = 0
    window_seconds = 0.0
    for step in range(start, end):
        began = time.perf_counter()
        stats = sft_step(
            model, optimizer, train_shard, train_config, step=step, device=device, amp=amp
        )
        window_seconds += time.perf_counter() - began
        window_tokens += stats.tokens
        train_loss = stats.loss
        completed = step + 1
        last = completed == end
        if completed % train_config.log_every == 0 or last:
            log(
                {
                    "event": "train",
                    "step": completed,
                    "loss": round(stats.loss, 4),
                    "lr": stats.lr,
                    "grad_norm": round(stats.grad_norm, 4),
                    "trained_tokens": stats.trained_tokens,
                    "tokens_per_sec": round(window_tokens / max(window_seconds, 1e-9)),
                    "peak_memory_mb": peak_memory_mb(device),
                }
            )
            window_tokens = 0
            window_seconds = 0.0
        if completed % train_config.eval_every == 0 or last:
            val_loss = sft_evaluate(model, val_shard, train_config, device=device, amp=amp)
            if not math.isfinite(val_loss):
                raise TrainError(f"validation loss is {val_loss} at step {completed}")
            log({"event": "eval", "step": completed, "val_loss": round(val_loss, 4)})
        if completed % train_config.checkpoint_every == 0 or last:
            save_checkpoint(checkpoint_path, model=model, optimizer=optimizer, step=completed)
            log({"event": "checkpoint", "step": completed})
    return SftResult(step=end, train_loss=train_loss, val_loss=val_loss)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="blom.sft", description="Fine-tune a pretrained Blom model on chat data."
    )
    parser.add_argument("--config", type=Path, required=True, help="TOML with a [train] table")
    parser.add_argument("--init", type=Path, help="pretrained checkpoint to start from")
    parser.add_argument("--data", type=Path, required=True, help="directory made by tokenize-chat")
    parser.add_argument("--out", type=Path, required=True, help="run directory")
    parser.add_argument("--until", type=int, help="stop after this global step")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        device = resolve_device(args.device)
        amp = device.type == "cuda" if args.amp is None else args.amp
        train_config = TrainConfig.load(args.config)
        args.out.mkdir(parents=True, exist_ok=True)
        with (args.out / METRICS_NAME).open("a", encoding="utf-8") as metrics:

            def log(record: dict[str, object]) -> None:
                line = json.dumps(record)
                print(line, flush=True)
                metrics.write(line + "\n")
                metrics.flush()

            sft_run(
                train_config,
                init=args.init,
                data_dir=args.data,
                out_dir=args.out,
                device=device,
                amp=amp,
                log=log,
                until=args.until,
            )
    except (TrainConfigError, DataError, CheckpointError, TrainError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())