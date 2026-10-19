import json
from pathlib import Path

import numpy as np
import pytest
import torch

from blom.data import DataError, Dataset, Shard, eval_batches, train_batch


def write_shard(path: Path, tokens: list[int] | np.ndarray) -> Shard:
    np.asarray(tokens, dtype="<u2").tofile(path)
    return Shard(path)


def counting_shard(tmp_path: Path, length: int) -> Shard:
    return write_shard(tmp_path / "count.bin", np.arange(length))


def test_shard_reads_back_what_was_written(tmp_path: Path) -> None:
    shard = write_shard(tmp_path / "a.bin", [5, 6, 7, 65535, 1])
    assert len(shard) == 5
    assert shard.window(1, 3).tolist() == [6, 7, 65535]
    assert shard.window(0, 5).dtype == np.int64


@pytest.mark.parametrize(("start", "length"), [(-1, 2), (0, 0), (4, 2), (6, 1)])
def test_window_rejects_out_of_range_requests(tmp_path: Path, start: int, length: int) -> None:
    shard = write_shard(tmp_path / "a.bin", [1, 2, 3, 4, 5])
    with pytest.raises(DataError, match="outside"):
        shard.window(start, length)


def test_shard_rejects_missing_empty_and_odd_files(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="cannot read"):
        Shard(tmp_path / "missing.bin")
    (tmp_path / "empty.bin").write_bytes(b"")
    with pytest.raises(DataError, match="not a valid"):
        Shard(tmp_path / "empty.bin")
    (tmp_path / "odd.bin").write_bytes(b"\x01\x02\x03")
    with pytest.raises(DataError, match="not a valid"):
        Shard(tmp_path / "odd.bin")


def test_max_token_scans_across_chunk_boundaries(tmp_path: Path) -> None:
    tokens = [1] * 25
    tokens[19] = 900
    shard = write_shard(tmp_path / "a.bin", tokens)
    assert shard.max_token(chunk=4) == 900


def test_train_batch_targets_are_inputs_shifted_by_one(tmp_path: Path) -> None:
    shard = counting_shard(tmp_path, 1000)
    inputs, targets = train_batch(shard, batch_size=6, seq_len=16, seed=1, step=0)
    assert inputs.shape == targets.shape == (6, 16)
    assert inputs.dtype == targets.dtype == torch.int64
    assert torch.equal(targets, inputs + 1)
    assert torch.equal(inputs[:, 1:], targets[:, :-1])
    assert torch.equal(inputs, inputs[:, :1] + torch.arange(16))


def test_train_batch_depends_only_on_seed_and_step(tmp_path: Path) -> None:
    shard = counting_shard(tmp_path, 1000)
    first = train_batch(shard, batch_size=4, seq_len=8, seed=7, step=3)
    for step in (0, 1, 2):
        train_batch(shard, batch_size=4, seq_len=8, seed=7, step=step)
    again = train_batch(shard, batch_size=4, seq_len=8, seed=7, step=3)
    assert torch.equal(first[0], again[0])
    other_step = train_batch(shard, batch_size=4, seq_len=8, seed=7, step=4)
    other_seed = train_batch(shard, batch_size=4, seq_len=8, seed=8, step=3)
    assert not torch.equal(first[0], other_step[0])
    assert not torch.equal(first[0], other_seed[0])


def test_train_batch_can_reach_the_very_last_token(tmp_path: Path) -> None:
    shard = counting_shard(tmp_path, 10)
    seen_last = False
    for step in range(200):
        _, targets = train_batch(shard, batch_size=4, seq_len=8, seed=0, step=step)
        assert int(targets.max()) <= 9
        seen_last = seen_last or int(targets.max()) == 9
    assert seen_last


def test_train_batch_rejects_a_shard_that_is_too_small(tmp_path: Path) -> None:
    shard = counting_shard(tmp_path, 8)
    with pytest.raises(DataError, match="need at least 9"):
        train_batch(shard, batch_size=1, seq_len=8, seed=0, step=0)


def test_eval_batches_cover_every_window_exactly_once(tmp_path: Path) -> None:
    shard = counting_shard(tmp_path, 100)
    batches = list(eval_batches(shard, batch_size=4, seq_len=10))
    assert [len(inputs) for inputs, _ in batches] == [4, 4, 1]
    targets = torch.cat([target for _, target in batches]).flatten()
    assert torch.equal(targets, torch.arange(1, 91))
    inputs = torch.cat([inputs for inputs, _ in batches]).flatten()
    assert torch.equal(inputs, torch.arange(0, 90))


def test_eval_batches_reject_a_shard_that_is_too_small(tmp_path: Path) -> None:
    shard = counting_shard(tmp_path, 10)
    with pytest.raises(DataError, match="need at least 11"):
        eval_batches(shard, batch_size=1, seq_len=10)


def write_dataset(directory: Path, *, vocab: int = 50, fmt: str = "u16-le", extra: int = 0) -> None:
    write_shard(directory / "train.bin", [1, 2, 3, 4, 5, 6, 7, 8, 0])
    write_shard(directory / "val.bin", [9, 8, 7, 0])
    meta = {
        "format": fmt,
        "vocab_size": vocab,
        "end_of_text_id": 0,
        "train": {"documents": 1, "tokens": 9 + extra},
        "val": {"documents": 1, "tokens": 4},
    }
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_dataset_opens_a_consistent_directory(tmp_path: Path) -> None:
    write_dataset(tmp_path)
    dataset = Dataset.open(tmp_path, expected_vocab=50)
    assert len(dataset.train) == 9
    assert len(dataset.val) == 4
    assert dataset.vocab_size == 50
    assert dataset.end_of_text_id == 0


def test_dataset_rejects_a_vocabulary_mismatch(tmp_path: Path) -> None:
    write_dataset(tmp_path)
    with pytest.raises(DataError, match="vocab 50, model expects 64"):
        Dataset.open(tmp_path, expected_vocab=64)


def test_dataset_rejects_token_ids_outside_the_vocabulary(tmp_path: Path) -> None:
    write_dataset(tmp_path, vocab=8)
    with pytest.raises(DataError, match="outside the vocabulary"):
        Dataset.open(tmp_path)
    assert Dataset.open(tmp_path, verify_tokens=False).vocab_size == 8


def test_dataset_rejects_a_truncated_shard(tmp_path: Path) -> None:
    write_dataset(tmp_path, extra=2)
    with pytest.raises(DataError, match=r"train\.bin has 9 tokens"):
        Dataset.open(tmp_path)


def test_dataset_rejects_unknown_formats_and_missing_meta(tmp_path: Path) -> None:
    write_dataset(tmp_path, fmt="u32-be")
    with pytest.raises(DataError, match="unsupported shard format"):
        Dataset.open(tmp_path)
    (tmp_path / "meta.json").unlink()
    with pytest.raises(DataError, match=r"cannot read meta\.json"):
        Dataset.open(tmp_path)
