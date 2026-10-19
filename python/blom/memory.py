import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from blom.data import DataError, Dataset, Shard, train_starts

FORMAT = "sketch-v1"
ABSENT = -1
MASK = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15
MIX_A = 0xBF58476D1CE4E5B9
MIX_B = 0x94D049BB133111EB


class SketchError(ValueError):
    pass


class SplitMix64:
    def __init__(self, seed: int) -> None:
        self._state = seed & MASK

    def next(self) -> int:
        self._state = (self._state + GAMMA) & MASK
        z = self._state
        z = ((z ^ (z >> 30)) * MIX_A) & MASK
        z = ((z ^ (z >> 27)) * MIX_B) & MASK
        return z ^ (z >> 31)


class NgramHasher:
    def __init__(self, seed: int, rows: int, orders: Sequence[int], n_hash: int) -> None:
        if rows <= 0 or n_hash <= 0 or not orders or min(orders) <= 0:
            raise SketchError("rows, orders and hashes must be positive")
        rng = SplitMix64(seed)
        self._salts: list[int] = []
        self._mults: list[tuple[int, ...]] = []
        for order in orders:
            for _ in range(n_hash):
                self._salts.append(rng.next() >> 33)
                self._mults.append(tuple((rng.next() >> 33) | 1 for _ in range(order)))
        self.rows = rows
        self.orders = tuple(orders)
        self.n_hash = n_hash
        self.lookup_count = len(self._salts)
        self.history = max(self.orders) - 1

    def rows_for(self, tokens: np.ndarray) -> np.ndarray:
        if tokens.ndim != 2:
            raise SketchError("tokens must be a 2-D array")
        batch, width = tokens.shape
        if width <= self.history:
            raise SketchError(f"need more than {self.history} tokens per row, got {width}")
        if tokens.size and int(tokens.min()) < ABSENT:
            raise SketchError("token ids must not be below -1")
        shifted = (tokens.astype(np.int64) + 1).astype(np.uint64)
        positions = width - self.history
        modulus = np.uint64(self.rows)
        out = np.empty((batch, positions, self.lookup_count), dtype=np.int64)
        for lookup, (salt, mults) in enumerate(zip(self._salts, self._mults, strict=True)):
            acc = np.full((batch, positions), salt, dtype=np.uint64)
            for back, mult in enumerate(mults):
                acc += shifted[:, self.history - back : width - back] * np.uint64(mult)
            out[:, :, lookup] = (acc % modulus).astype(np.int64)
        return out


def window_with_history(shard: Shard, start: int, length: int, history: int) -> np.ndarray:
    first = max(0, start - history)
    available = start - first
    body = shard.window(first, available + length)
    missing = history - available
    if missing > 0:
        return np.concatenate([np.full(missing, ABSENT, dtype=np.int64), body])
    return body


def _require_int(manifest: dict[str, object], key: str) -> int:
    value = manifest.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise SketchError(f"sketch.json needs an integer {key}")
    return value


def _require_orders(manifest: dict[str, object]) -> list[int]:
    value = manifest.get("orders")
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    ):
        raise SketchError("sketch.json needs a list of integer orders")
    return [int(item) for item in value]


def _file_name(manifest: dict[str, object], key: str) -> str:
    files = manifest.get("files")
    name = files.get(key) if isinstance(files, dict) else None
    if not isinstance(name, str) or Path(name).name != name or not name:
        raise SketchError(f"sketch.json needs a plain file name for {key}")
    return name


def _checked_size(path: Path, expected: int) -> None:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise SketchError(f"cannot read {path}: {error}") from error
    if size != expected:
        raise SketchError(f"{path} has {size} bytes, expected {expected}")


class SketchReader:
    def __init__(self, directory: Path, *, in_memory: bool = True) -> None:
        try:
            manifest = json.loads((directory / "sketch.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SketchError(f"cannot read sketch.json in {directory}: {error}") from error
        if not isinstance(manifest, dict):
            raise SketchError("sketch.json must contain an object")
        if manifest.get("format") != FORMAT:
            raise SketchError(f"unsupported sketch format {manifest.get('format')!r}")
        self.directory = directory
        self.seed = _require_int(manifest, "seed")
        self.rows = _require_int(manifest, "rows")
        self.n_hash = _require_int(manifest, "hashes")
        self.part_dim = _require_int(manifest, "part_dim")
        self.vocab_size = _require_int(manifest, "vocab_size")
        self.pairs = _require_int(manifest, "pairs")
        self.source_tokens = _require_int(manifest, "source_tokens")
        self.orders = _require_orders(manifest)
        if self.part_dim <= 0 or self.vocab_size <= 0:
            raise SketchError("part_dim and vocab_size must be positive")
        self.hasher = NgramHasher(self.seed, self.rows, self.orders, self.n_hash)
        if _require_int(manifest, "lookups") != self.hasher.lookup_count:
            raise SketchError("sketch.json lookups do not match orders and hashes")

        projection_path = directory / _file_name(manifest, "projection")
        sums_path = directory / _file_name(manifest, "sums")
        counts_path = directory / _file_name(manifest, "counts")
        _checked_size(projection_path, self.vocab_size * self.part_dim * 4)
        _checked_size(sums_path, self.rows * self.part_dim * 4)
        _checked_size(counts_path, self.rows * 4)
        self.projection = np.fromfile(projection_path, dtype="<f4").reshape(
            self.vocab_size, self.part_dim
        )
        self.counts = np.fromfile(counts_path, dtype="<u4")
        total = int(self.counts.sum(dtype=np.uint64))
        if total != self.pairs * self.hasher.lookup_count:
            raise SketchError("counts.u32 does not add up to pairs times lookups")
        self.sums: np.ndarray
        if in_memory:
            self.sums = np.fromfile(sums_path, dtype="<f4").reshape(self.rows, self.part_dim)
        else:
            self.sums = np.memmap(
                sums_path, dtype="<f4", mode="r", shape=(self.rows, self.part_dim)
            )

    @property
    def lookups(self) -> int:
        return self.hasher.lookup_count

    @property
    def history(self) -> int:
        return self.hasher.history

    @property
    def feature_dim(self) -> int:
        return self.lookups * self.part_dim

    def fingerprint(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "rows": self.rows,
            "orders": list(self.orders),
            "hashes": self.n_hash,
            "part_dim": self.part_dim,
            "vocab_size": self.vocab_size,
            "pairs": self.pairs,
            "source_tokens": self.source_tokens,
        }

    def check_dataset(self, dataset: Dataset) -> None:
        if dataset.vocab_size != self.vocab_size:
            raise SketchError(
                f"sketch has vocab {self.vocab_size}, data has vocab {dataset.vocab_size}"
            )
        if len(dataset.train) != self.source_tokens:
            raise SketchError(
                f"sketch was built from {self.source_tokens} training tokens, "
                f"data has {len(dataset.train)}"
            )

    def features(self, tokens: np.ndarray, *, targets: np.ndarray | None = None) -> np.ndarray:
        rows = self.hasher.rows_for(tokens)
        batch, positions, lookups = rows.shape
        sums = np.asarray(self.sums[rows], dtype=np.float32)
        counts = self.counts[rows].astype(np.float32)
        if targets is not None:
            if targets.shape != (batch, positions):
                raise SketchError(f"targets must have shape {(batch, positions)}")
            if targets.size and (int(targets.min()) < 0 or int(targets.max()) >= self.vocab_size):
                raise SketchError("targets contain ids outside the vocabulary")
            own = self.projection[targets]
            shared = (rows[:, :, :, None] == rows[:, :, None, :]).sum(axis=3).astype(np.float32)
            sums = sums - shared[:, :, :, None] * own[:, :, None, :]
            counts = counts - shared
            if counts.size and float(counts.min()) < 0:
                raise SketchError(
                    "leave-one-out needs windows taken from the sketched training data"
                )
        features = sums / (counts[:, :, :, None] + 1.0)
        return features.reshape(batch, positions, lookups * self.part_dim)


def _batch_with_memory(
    shard: Shard,
    reader: SketchReader,
    starts: Sequence[int],
    seq_len: int,
    *,
    leave_one_out: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    history = reader.history
    ext = np.stack([window_with_history(shard, start, seq_len + 1, history) for start in starts])
    window = ext[:, history:]
    targets = window[:, 1:]
    mem = reader.features(ext[:, :-1], targets=targets if leave_one_out else None)
    return (
        torch.from_numpy(np.ascontiguousarray(window[:, :-1])),
        torch.from_numpy(np.ascontiguousarray(targets)),
        torch.from_numpy(mem),
    )


def train_batch_with_memory(
    shard: Shard, reader: SketchReader, *, batch_size: int, seq_len: int, seed: int, step: int
) -> tuple[Tensor, Tensor, Tensor]:
    starts = train_starts(shard, batch_size=batch_size, seq_len=seq_len, seed=seed, step=step)
    return _batch_with_memory(shard, reader, starts, seq_len, leave_one_out=True)


def eval_batches_with_memory(
    shard: Shard, reader: SketchReader, *, batch_size: int, seq_len: int
) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
    windows = (len(shard) - 1) // seq_len
    if windows == 0:
        raise DataError(f"shard has {len(shard)} tokens, need at least {seq_len + 1}")
    for first in range(0, windows, batch_size):
        last = min(first + batch_size, windows)
        starts = [index * seq_len for index in range(first, last)]
        yield _batch_with_memory(shard, reader, starts, seq_len, leave_one_out=False)


def verify(
    reader: SketchReader, shard: Shard, *, sample_rows: int = 64, chunk: int = 1 << 21
) -> tuple[bool, int]:
    if len(shard) != reader.source_tokens:
        raise SketchError("the shard is not the one the sketch was built from")
    history = reader.history
    rng = np.random.default_rng(0)
    used = np.flatnonzero(reader.counts)
    watched = np.sort(rng.choice(used, size=min(sample_rows, used.size), replace=False))
    slot = np.full(reader.rows, -1, dtype=np.int64)
    slot[watched] = np.arange(watched.size)
    expected = np.zeros((watched.size, reader.part_dim), dtype=np.float64)
    seen = np.zeros(reader.rows, dtype=np.int64)
    pairs = len(shard) - 1
    for start in range(0, pairs, chunk):
        length = min(chunk, pairs - start)
        ext = window_with_history(shard, start, length + 1, history)
        rows = reader.hasher.rows_for(ext[None, :-1])[0]
        following = ext[history + 1 :]
        seen += np.bincount(rows.ravel(), minlength=reader.rows)
        hit_slots = slot[rows]
        hits = hit_slots >= 0
        if hits.any():
            token_grid = np.broadcast_to(following[:, None], rows.shape)
            np.add.at(
                expected,
                hit_slots[hits],
                reader.projection[token_grid[hits]].astype(np.float64),
            )
    counts_match = bool(np.array_equal(seen, reader.counts.astype(np.int64)))
    stored = np.asarray(reader.sums[watched], dtype=np.float64)
    sums_match = bool(np.allclose(stored, expected, rtol=1e-5, atol=1e-4))
    return counts_match and sums_match, int(watched.size)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m blom.memory", description="Check a sketch against its training shard"
    )
    parser.add_argument("--sketch", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sample-rows", type=int, default=64)
    args = parser.parse_args(argv)
    try:
        reader = SketchReader(args.sketch, in_memory=False)
        dataset = Dataset.open(args.data, verify_tokens=False)
        reader.check_dataset(dataset)
        ok, sampled = verify(reader, dataset.train, sample_rows=args.sample_rows)
    except (SketchError, DataError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        f"rows {reader.rows} | lookups {reader.lookups} | part_dim {reader.part_dim} | "
        f"sampled sum rows {sampled}"
    )
    if not ok:
        print("error: the sketch does not match the training shard", file=sys.stderr)
        return 1
    print("sketch matches the training shard: all counts and sampled sums agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
