import argparse
import json
import math
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from blom.checkpoint import CheckpointError, load_checkpoint, restore, save_checkpoint
from blom.config import ConfigError, ModelConfig
from blom.data import DataError, Dataset
from blom.model import GPT
from blom.optim import build_optimizer
from blom.train import TrainError, evaluate, train_step
from blom.train_config import TrainConfig, TrainConfigError

CHECKPOINT_NAME = "last.pt"
RUN_NAME = "run.json"
METRICS_NAME = "metrics.jsonl"
MEGABYTE = 1 << 20

Log = Callable[[dict[str, object]], None]


@dataclass(frozen=True, slots=True)
class RunResult:
    step: int
    train_loss: float | None
    val_loss: float | None


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise TrainError("cuda was requested but no CUDA device is available")
    return torch.device(name)


def check_run_config(out_dir: Path, model_config: ModelConfig, train_config: TrainConfig) -> None:
    description = {"model": asdict(model_config), "train": asdict(train_config)}
    path = out_dir / RUN_NAME
    if not path.exists():
        path.write_text(json.dumps(description, indent=2), encoding="utf-8")
        return
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrainError(f"cannot read {path}: {error}") from error
    if previous != description:
        raise TrainError(f"{path} describes a different configuration; use a new output directory")


def peak_memory_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return round(torch.cuda.max_memory_allocated(device) / MEGABYTE, 1)


def run(
    model_config: ModelConfig,
    train_config: TrainConfig,
    *,
    data_dir: Path,
    out_dir: Path,
    device: torch.device,
    amp: bool,
    log: Log,
    until: int | None = None,
) -> RunResult:
    end = train_config.total_steps if until is None else until
    if not 0 < end <= train_config.total_steps:
        raise TrainError(f"until must be in 1..{train_config.total_steps}, got {end}")
    if train_config.seq_len > model_config.max_seq:
        raise TrainError(
            f"seq_len {train_config.seq_len} exceeds the model's max_seq {model_config.max_seq}"
        )
    dataset = Dataset.open(data_dir, expected_vocab=model_config.vocab_size)
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
    log(
        {
            "event": "start",
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "amp": amp,
            "parameters": model_config.param_count,
            "tokens_per_step": train_config.tokens_per_step,
            "total_steps": train_config.total_steps,
            "until": end,
        }
    )
    checkpoint_path = out_dir / CHECKPOINT_NAME
    start = 0
    if checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path, map_location=device)
        start = restore(checkpoint, model=model, optimizer=optimizer)
        log({"event": "resume", "step": start})
    if start >= end:
        log({"event": "nothing_to_do", "step": start, "until": end})
        return RunResult(step=start, train_loss=None, val_loss=None)

    train_loss: float | None = None
    val_loss: float | None = None
    window_tokens = 0
    window_seconds = 0.0
    for step in range(start, end):
        began = time.perf_counter()
        stats = train_step(
            model, optimizer, dataset.train, train_config, step=step, device=device, amp=amp
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
                    "tokens_per_sec": round(window_tokens / max(window_seconds, 1e-9)),
                    "peak_memory_mb": peak_memory_mb(device),
                }
            )
            window_tokens = 0
            window_seconds = 0.0
        if completed % train_config.eval_every == 0 or last:
            val_loss = evaluate(model, dataset.val, train_config, device=device, amp=amp)
            if not math.isfinite(val_loss):
                raise TrainError(f"validation loss is {val_loss} at step {completed}")
            log({"event": "eval", "step": completed, "val_loss": round(val_loss, 4)})
        if completed % train_config.checkpoint_every == 0 or last:
            save_checkpoint(checkpoint_path, model=model, optimizer=optimizer, step=completed)
            log({"event": "checkpoint", "step": completed})
    return RunResult(step=end, train_loss=train_loss, val_loss=val_loss)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="blom.run", description="Train a Blom model.")
    parser.add_argument("--config", type=Path, required=True, help="TOML with [model] and [train]")
    parser.add_argument("--data", type=Path, required=True, help="directory made by dataprep")
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
        model_config = ModelConfig.load(args.config)
        train_config = TrainConfig.load(args.config)
        args.out.mkdir(parents=True, exist_ok=True)
        with (args.out / METRICS_NAME).open("a", encoding="utf-8") as metrics:

            def log(record: dict[str, object]) -> None:
                line = json.dumps(record)
                print(line, flush=True)
                metrics.write(line + "\n")
                metrics.flush()

            run(
                model_config,
                train_config,
                data_dir=args.data,
                out_dir=args.out,
                device=device,
                amp=amp,
                log=log,
                until=args.until,
            )
    except (
        ConfigError,
        TrainConfigError,
        DataError,
        CheckpointError,
        TrainError,
        OSError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
