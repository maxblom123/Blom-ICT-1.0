import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

TOKEN_DTYPE = np.dtype("<u2")
BYTES_PER_TOKEN = TOKEN_DTYPE.itemsize
SCAN_CHUNK = 1 << 24


class DataError(ValueError):
    pass


class Shard:
    def __init__(self, path: Path) -> None:
        try:
            size = path.stat().st_size
        except OSError as error:
            raise DataError(f"cannot read shard {path}: {error}") from error
        if size == 0 or size % BYTES_PER_TOKEN != 0:
            raise DataError(f"{path} is not a valid token shard ({size} bytes)")
        self.path = path
        self._length = size // BYTES_PER_TOKEN

    def __len__(self) -> int:
        return self._length

    def window(self, start: int, length: int) -> np.ndarray:
        if start < 0 or length <= 0 or start + length > self._length:
            raise DataError(f"window {start}+{length} is outside a shard of {self._length} tokens")
        tokens = np.fromfile(
            self.path, dtype=TOKEN_DTYPE, count=length, offset=start * BYTES_PER_TOKEN
        )
        return tokens.astype(np.int64)

    def max_token(self, chunk: int = SCAN_CHUNK) -> int:
        largest = 0
        for start in range(0, self._length, chunk):
            largest = max(largest, int(self.window(start, min(chunk, self._length - start)).max()))
        return largest


def _pair(rows: list[np.ndarray]) -> tuple[Tensor, Tensor]:
    batch = torch.from_numpy(np.stack(rows))
    return batch[:, :-1].contiguous(), batch[:, 1:].contiguous()


def train_starts(shard: Shard, *, batch_size: int, seq_len: int, seed: int, step: int) -> list[int]:
    if len(shard) < seq_len + 1:
        raise DataError(f"shard has {len(shard)} tokens, need at least {seq_len + 1}")
    rng = np.random.default_rng((seed, step))
    return [int(start) for start in rng.integers(0, len(shard) - seq_len, size=batch_size)]


def train_batch(
    shard: Shard, *, batch_size: int, seq_len: int, seed: int, step: int
) -> tuple[Tensor, Tensor]:
    starts = train_starts(shard, batch_size=batch_size, seq_len=seq_len, seed=seed, step=step)
    return _pair([shard.window(start, seq_len + 1) for start in starts])


def eval_batches(shard: Shard, *, batch_size: int, seq_len: int) -> Iterator[tuple[Tensor, Tensor]]:
    windows = (len(shard) - 1) // seq_len
    if windows == 0:
        raise DataError(f"shard has {len(shard)} tokens, need at least {seq_len + 1}")
    return _eval_batches(shard, batch_size, seq_len, windows)


def _eval_batches(
    shard: Shard, batch_size: int, seq_len: int, windows: int
) -> Iterator[tuple[Tensor, Tensor]]:
    for first in range(0, windows, batch_size):
        last = min(first + batch_size, windows)
        yield _pair([shard.window(index * seq_len, seq_len + 1) for index in range(first, last)])


def _count(meta: dict[str, object], part: str) -> int:
    table = meta.get(part)
    tokens = table.get("tokens") if isinstance(table, dict) else None
    if not isinstance(tokens, int) or isinstance(tokens, bool):
        raise DataError(f"meta.json has no integer {part}.tokens")
    return tokens


@dataclass(frozen=True, slots=True)
class Dataset:
    train: Shard
    val: Shard
    vocab_size: int
    end_of_text_id: int

    @classmethod
    def open(
        cls, directory: Path, *, expected_vocab: int | None = None, verify_tokens: bool = True
    ) -> "Dataset":
        try:
            meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise DataError(f"cannot read meta.json in {directory}: {error}") from error
        if not isinstance(meta, dict):
            raise DataError("meta.json must contain an object")
        if meta.get("format") != "u16-le":
            raise DataError(f"unsupported shard format {meta.get('format')!r}")
        vocab = meta.get("vocab_size")
        end_id = meta.get("end_of_text_id")
        if not isinstance(vocab, int) or not isinstance(end_id, int):
            raise DataError("meta.json needs integer vocab_size and end_of_text_id")
        if expected_vocab is not None and vocab != expected_vocab:
            raise DataError(f"data has vocab {vocab}, model expects {expected_vocab}")
        train = Shard(directory / "train.bin")
        val = Shard(directory / "val.bin")
        for name, shard in (("train", train), ("val", val)):
            if len(shard) != _count(meta, name):
                raise DataError(f"{name}.bin has {len(shard)} tokens, meta.json says otherwise")
            if verify_tokens and shard.max_token() >= vocab:
                raise DataError(f"{name}.bin contains token ids outside the vocabulary")
        return cls(train=train, val=val, vocab_size=vocab, end_of_text_id=end_id)
