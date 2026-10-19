import json
from pathlib import Path

import numpy as np
import pytest

from blom.data import Dataset, Shard
from blom.memory import (
    ABSENT,
    NgramHasher,
    SketchError,
    SketchReader,
    SplitMix64,
    main,
    verify,
    window_with_history,
)

GOLDEN = [
    (
        (20_261_016, 8_388_593, [2, 3], 2),
        [
            ([0], [5269109, 33755, 6339393, 2752207]),
            ([5], [8332236, 2598646, 2649679, 2427908]),
            ([5, 17], [6929112, 5022148, 7126768, 3567753]),
            ([5, 17, 99], [2745328, 4389797, 2174190, 6909706]),
            ([8191, 8191, 8191], [2630494, 5348596, 4884873, 2599925]),
            ([0, 0, 0], [454567, 6122590, 794225, 1079725]),
            ([1, 2, 3, 4], [3560910, 4630666, 3234793, 870833]),
            ([4000, 12, 7, 8190], [1330557, 5407835, 7257528, 3254947]),
            ([123, 456, 789, 1011, 1213], [8288127, 4516469, 295807, 2541637]),
        ],
    ),
    (
        (1, 1_000_003, [1, 2, 4], 1),
        [
            ([0], [227393, 457571, 384624]),
            ([5], [974017, 714023, 816159]),
            ([5, 17], [565908, 19417, 500910]),
            ([5, 17, 99], [610505, 205044, 616555]),
            ([8191, 8191, 8191], [743162, 604038, 679050]),
            ([0, 0, 0], [227393, 505890, 782862]),
            ([1, 2, 3, 4], [424691, 656008, 378630]),
            ([4000, 12, 7, 8190], [193836, 911239, 576223]),
            ([123, 456, 789, 1011, 1213], [557833, 171320, 649968]),
        ],
    ),
]

SEED = 5
ROWS = 1_000_003
VOCAB = 10
PART_DIM = 3


def stream(length: int, vocab: int = VOCAB, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, vocab, size=length).astype(np.int64)


def extended(tokens: np.ndarray, history: int) -> np.ndarray:
    return np.concatenate([np.full(history, ABSENT, dtype=np.int64), tokens])[None, :]


def write_stream(path: Path, tokens: np.ndarray) -> Shard:
    tokens.astype("<u2").tofile(path)
    return Shard(path)


def reference_table(
    hasher: NgramHasher,
    projection: np.ndarray,
    tokens: np.ndarray,
    skip: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = hasher.rows_for(extended(tokens, hasher.history)[:, :-1])[0]
    sums = np.zeros((hasher.rows, projection.shape[1]), dtype=np.float64)
    counts = np.zeros(hasher.rows, dtype=np.int64)
    for position in range(len(tokens) - 1):
        if position == skip:
            continue
        for lookup in range(hasher.lookup_count):
            row = rows[position, lookup]
            sums[row] += projection[tokens[position + 1]]
            counts[row] += 1
    return sums, counts, rows


def write_sketch(
    directory: Path,
    tokens: np.ndarray,
    *,
    rows: int = ROWS,
    orders: tuple[int, ...] = (2, 3),
    n_hash: int = 2,
    vocab: int = VOCAB,
    part_dim: int = PART_DIM,
    overrides: dict[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    directory.mkdir(parents=True, exist_ok=True)
    hasher = NgramHasher(SEED, rows, orders, n_hash)
    projection = (
        np.random.default_rng(1).standard_normal((vocab, part_dim)).astype(np.float32)
        / np.sqrt(part_dim)
    ).astype(np.float32)
    sums, counts, _ = reference_table(hasher, projection, tokens)
    projection.astype("<f4").tofile(directory / "projection.f32")
    sums.astype("<f4").tofile(directory / "sums.f32")
    counts.astype("<u4").tofile(directory / "counts.u32")
    manifest: dict[str, object] = {
        "format": "sketch-v1",
        "seed": SEED,
        "rows": rows,
        "orders": list(orders),
        "hashes": n_hash,
        "part_dim": part_dim,
        "vocab_size": vocab,
        "lookups": hasher.lookup_count,
        "pairs": len(tokens) - 1,
        "source_tokens": len(tokens),
        "rows_used": int((counts > 0).sum()),
        "max_count": int(counts.max()),
        "files": {
            "projection": "projection.f32",
            "sums": "sums.f32",
            "counts": "counts.u32",
        },
    }
    manifest.update(overrides or {})
    (directory / "sketch.json").write_text(json.dumps(manifest), encoding="utf-8")
    return projection, sums, counts


def write_data(directory: Path, train: np.ndarray, val: np.ndarray, vocab: int = VOCAB) -> Dataset:
    directory.mkdir(parents=True, exist_ok=True)
    train.astype("<u2").tofile(directory / "train.bin")
    val.astype("<u2").tofile(directory / "val.bin")
    meta = {
        "format": "u16-le",
        "vocab_size": vocab,
        "end_of_text_id": 0,
        "train": {"tokens": len(train)},
        "val": {"tokens": len(val)},
    }
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return Dataset.open(directory)


def test_splitmix64_matches_the_rust_reference_values() -> None:
    rng = SplitMix64(0)
    assert rng.next() == 0xE220A8397B1DCDAF
    assert rng.next() == 0x6E789E6AA1B965F4
    assert rng.next() == 0x06C45D188009454F


@pytest.mark.parametrize(("config", "cases"), GOLDEN)
def test_rows_match_the_rust_hasher_exactly(
    config: tuple[int, int, list[int], int], cases: list[tuple[list[int], list[int]]]
) -> None:
    seed, rows, orders, n_hash = config
    hasher = NgramHasher(seed, rows, orders, n_hash)
    width = hasher.history + 1
    for context, expected in cases:
        padded = ([ABSENT] * width + context)[-width:]
        got = hasher.rows_for(np.array([padded], dtype=np.int64))
        assert got.shape == (1, 1, hasher.lookup_count)
        assert got[0, 0].tolist() == expected, context


def test_rows_have_the_documented_shape_and_range() -> None:
    hasher = NgramHasher(3, 4093, [2, 3], 2)
    tokens = stream(40, vocab=100).reshape(4, 10)
    rows = hasher.rows_for(tokens)
    assert rows.shape == (4, 10 - hasher.history, 4)
    assert rows.dtype == np.int64
    assert rows.min() >= 0
    assert rows.max() < 4093
    assert hasher.lookup_count == 4
    assert hasher.history == 2


def test_only_the_last_order_tokens_matter() -> None:
    hasher = NgramHasher(7, 65_537, [2], 1)
    first = hasher.rows_for(np.array([[9, 9, 4, 8]]))
    second = hasher.rows_for(np.array([[1, 2, 4, 8]]))
    third = hasher.rows_for(np.array([[9, 9, 5, 8]]))
    assert first[0, -1, 0] == second[0, -1, 0]
    assert first[0, -1, 0] != third[0, -1, 0]


def test_before_the_start_differs_from_token_zero() -> None:
    hasher = NgramHasher(7, 65_537, [2], 1)
    absent = hasher.rows_for(np.array([[ABSENT, 3]]))
    zero = hasher.rows_for(np.array([[0, 3]]))
    assert absent[0, 0, 0] != zero[0, 0, 0]


def test_hasher_rejects_bad_configurations_and_inputs() -> None:
    for bad in [(0, [2], 1), (8, [], 1), (8, [0], 1), (8, [2], 0), (-1, [2], 1)]:
        with pytest.raises(SketchError, match="positive"):
            NgramHasher(1, bad[0], bad[1], bad[2])
    hasher = NgramHasher(1, 101, [3], 1)
    with pytest.raises(SketchError, match="2-D"):
        hasher.rows_for(np.array([1, 2, 3, 4]))
    with pytest.raises(SketchError, match="more than 2"):
        hasher.rows_for(np.array([[1, 2]]))
    with pytest.raises(SketchError, match="below -1"):
        hasher.rows_for(np.array([[1, 2, -2, 4]]))


def test_window_with_history_pads_only_before_the_start_of_the_stream(tmp_path: Path) -> None:
    shard = write_stream(tmp_path / "a.bin", np.arange(10, 30))
    assert window_with_history(shard, 0, 4, 3).tolist() == [ABSENT] * 3 + [10, 11, 12, 13]
    assert window_with_history(shard, 1, 4, 3).tolist() == [ABSENT] * 2 + [10, 11, 12, 13, 14]
    assert window_with_history(shard, 3, 4, 3).tolist() == [10, 11, 12, 13, 14, 15, 16]
    assert window_with_history(shard, 10, 4, 3).tolist() == [17, 18, 19, 20, 21, 22, 23]
    assert window_with_history(shard, 5, 4, 0).tolist() == [15, 16, 17, 18]


def test_windows_with_history_give_the_same_rows_as_the_whole_stream(tmp_path: Path) -> None:
    tokens = stream(300, vocab=50)
    shard = write_stream(tmp_path / "a.bin", tokens)
    hasher = NgramHasher(11, 10_007, [1, 2, 3], 2)
    whole = hasher.rows_for(extended(tokens, hasher.history))[0]
    for start, length in [(0, 17), (1, 40), (2, 9), (50, 100), (299 - 20, 20)]:
        ext = window_with_history(shard, start, length, hasher.history)
        part = hasher.rows_for(ext[None, :])[0]
        assert np.array_equal(part, whole[start : start + length]), (start, length)


def test_reader_exposes_the_manifest_and_arrays(tmp_path: Path) -> None:
    tokens = stream(200)
    projection, sums, counts = write_sketch(tmp_path / "s", tokens)
    reader = SketchReader(tmp_path / "s")
    assert (reader.rows, reader.part_dim, reader.vocab_size) == (ROWS, PART_DIM, VOCAB)
    assert reader.orders == [2, 3]
    assert reader.lookups == 4
    assert reader.feature_dim == 12
    assert reader.history == 2
    assert reader.pairs == 199
    assert reader.source_tokens == 200
    assert np.array_equal(reader.projection, projection)
    assert np.array_equal(reader.counts, counts.astype(np.uint32))
    assert np.allclose(np.asarray(reader.sums[:5000]), sums[:5000].astype(np.float32))


def test_the_table_can_be_held_in_memory_or_mapped_with_the_same_result(tmp_path: Path) -> None:
    write_sketch(tmp_path / "s", stream(200))
    loaded = SketchReader(tmp_path / "s")
    mapped = SketchReader(tmp_path / "s", in_memory=False)
    assert not isinstance(loaded.sums, np.memmap)
    assert isinstance(mapped.sums, np.memmap)
    window = stream(30, seed=4).reshape(2, 15)
    assert np.array_equal(loaded.features(window), mapped.features(window))


def test_features_are_the_shrunk_mean_of_the_row(tmp_path: Path) -> None:
    tokens = stream(200)
    _, sums, counts = write_sketch(tmp_path / "s", tokens, rows=101)
    reader = SketchReader(tmp_path / "s")
    window = stream(30, seed=9).reshape(2, 15)
    rows = reader.hasher.rows_for(window)
    features = reader.features(window)
    assert features.shape == (2, 13, 12)
    assert features.dtype == np.float32
    for batch in range(2):
        for position in range(13):
            for lookup in range(4):
                row = rows[batch, position, lookup]
                count = counts[row]
                mean = sums[row] / count if count else np.zeros(PART_DIM)
                expected = mean * count / (count + 1)
                got = features[batch, position, lookup * PART_DIM : (lookup + 1) * PART_DIM]
                assert np.allclose(got, expected, atol=1e-5)


def test_unseen_contexts_give_zero_features(tmp_path: Path) -> None:
    write_sketch(tmp_path / "s", stream(20))
    reader = SketchReader(tmp_path / "s")
    window = np.array([[ABSENT, 9, 9, 9, 9, 9, 9, 9]])
    assert np.count_nonzero(reader.features(window)) == 0


@pytest.mark.parametrize(("rows", "orders", "n_hash"), [(7, (1, 2, 3), 2), (5, (2,), 3)])
def test_leave_one_out_equals_rebuilding_the_table_without_that_pair(
    tmp_path: Path, rows: int, orders: tuple[int, ...], n_hash: int
) -> None:
    tokens = stream(40)
    projection, _, _ = write_sketch(tmp_path / "s", tokens, rows=rows, orders=orders, n_hash=n_hash)
    reader = SketchReader(tmp_path / "s")
    ext = extended(tokens, reader.history)[:, :-1]
    got = reader.features(ext, targets=tokens[None, 1:])
    assert got.shape == (1, 39, reader.feature_dim)
    for position in range(39):
        sums, counts, table_rows = reference_table(reader.hasher, projection, tokens, skip=position)
        expected = np.concatenate(
            [
                sums[table_rows[position, lookup]] / (counts[table_rows[position, lookup]] + 1)
                for lookup in range(reader.lookups)
            ]
        )
        assert np.allclose(got[0, position], expected, atol=1e-5), position


def test_a_context_seen_once_has_no_memory_of_itself_under_leave_one_out(tmp_path: Path) -> None:
    tokens = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], dtype=np.int64)
    write_sketch(tmp_path / "s", tokens)
    reader = SketchReader(tmp_path / "s")
    window = extended(tokens, reader.history)[:, :-1]
    plain = reader.features(window)
    loo = reader.features(window, targets=tokens[None, 1:])
    assert np.count_nonzero(plain) > 0
    assert np.count_nonzero(loo) == 0


def test_leave_one_out_refuses_windows_that_are_not_in_the_sketch(tmp_path: Path) -> None:
    write_sketch(tmp_path / "s", stream(50))
    reader = SketchReader(tmp_path / "s")
    other = stream(20, seed=77)
    with pytest.raises(SketchError, match="training data"):
        reader.features(extended(other, 2)[:, :-1], targets=other[None, 1:])


def test_feature_targets_are_validated(tmp_path: Path) -> None:
    write_sketch(tmp_path / "s", stream(50))
    reader = SketchReader(tmp_path / "s")
    window = extended(stream(10), 2)[:, :-1]
    with pytest.raises(SketchError, match="shape"):
        reader.features(window, targets=np.zeros((1, 3), dtype=np.int64))
    with pytest.raises(SketchError, match="vocabulary"):
        reader.features(window, targets=np.full((1, 9), VOCAB, dtype=np.int64))
    with pytest.raises(SketchError, match="vocabulary"):
        reader.features(window, targets=np.full((1, 9), -1, dtype=np.int64))


def rewrite(directory: Path, change: dict[str, object]) -> None:
    path = directory / "sketch.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(change)
    path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"format": "sketch-v0"}, "unsupported"),
        ({"seed": "x"}, "integer seed"),
        ({"rows": True}, "integer rows"),
        ({"orders": []}, "orders"),
        ({"orders": [2, "3"]}, "orders"),
        ({"part_dim": 0}, "positive"),
        ({"lookups": 3}, "lookups"),
        ({"pairs": 5}, "add up"),
        ({"files": {"projection": "../x", "sums": "s", "counts": "c"}}, "plain file name"),
        ({"files": None}, "plain file name"),
        ({"part_dim": 4}, "bytes"),
    ],
)
def test_reader_rejects_inconsistent_manifests(
    tmp_path: Path, change: dict[str, object], message: str
) -> None:
    write_sketch(tmp_path / "s", stream(100))
    rewrite(tmp_path / "s", change)
    with pytest.raises(SketchError, match=message):
        SketchReader(tmp_path / "s")


def test_reader_rejects_missing_and_broken_files(tmp_path: Path) -> None:
    with pytest.raises(SketchError, match="cannot read"):
        SketchReader(tmp_path / "nothing")
    write_sketch(tmp_path / "s", stream(100))
    (tmp_path / "s" / "sketch.json").write_text("not json", encoding="utf-8")
    with pytest.raises(SketchError, match="cannot read"):
        SketchReader(tmp_path / "s")
    (tmp_path / "s" / "sketch.json").write_text("[]", encoding="utf-8")
    with pytest.raises(SketchError, match="object"):
        SketchReader(tmp_path / "s")
    write_sketch(tmp_path / "t", stream(100))
    (tmp_path / "t" / "counts.u32").unlink()
    with pytest.raises(SketchError, match="cannot read"):
        SketchReader(tmp_path / "t")
    write_sketch(tmp_path / "u", stream(100))
    (tmp_path / "u" / "sums.f32").write_bytes(b"\x00" * 8)
    with pytest.raises(SketchError, match="bytes"):
        SketchReader(tmp_path / "u")


def test_the_reader_is_checked_against_its_dataset(tmp_path: Path) -> None:
    train = stream(120)
    write_sketch(tmp_path / "s", train)
    reader = SketchReader(tmp_path / "s")
    reader.check_dataset(write_data(tmp_path / "ok", train, stream(30, seed=3)))
    with pytest.raises(SketchError, match="vocab"):
        reader.check_dataset(write_data(tmp_path / "v", train, stream(30, seed=3), vocab=11))
    with pytest.raises(SketchError, match="120 training tokens"):
        reader.check_dataset(write_data(tmp_path / "t", stream(100), stream(30, seed=3)))


def test_verify_accepts_a_correct_sketch_and_catches_corruption(tmp_path: Path) -> None:
    train = stream(300)
    write_sketch(tmp_path / "s", train)
    shard = write_stream(tmp_path / "train.bin", train)
    reader = SketchReader(tmp_path / "s")
    ok, sampled = verify(reader, shard, sample_rows=1000, chunk=64)
    assert ok
    assert sampled == int((reader.counts > 0).sum())

    counts = np.fromfile(tmp_path / "s" / "counts.u32", dtype="<u4")
    first, second = np.flatnonzero(counts)[:2]
    counts[first] += 1
    counts[second] -= 1
    counts.tofile(tmp_path / "s" / "counts.u32")
    assert not verify(SketchReader(tmp_path / "s"), shard, sample_rows=1000, chunk=64)[0]


def test_verify_catches_wrong_sums_and_the_wrong_shard(tmp_path: Path) -> None:
    train = stream(300)
    write_sketch(tmp_path / "s", train)
    shard = write_stream(tmp_path / "train.bin", train)
    sums = np.fromfile(tmp_path / "s" / "sums.f32", dtype="<f4")
    sums[np.flatnonzero(sums)[0]] += 0.5
    sums.tofile(tmp_path / "s" / "sums.f32")
    reader = SketchReader(tmp_path / "s")
    assert not verify(reader, shard, sample_rows=1000, chunk=64)[0]
    short = write_stream(tmp_path / "short.bin", train[:200])
    with pytest.raises(SketchError, match="not the one"):
        verify(reader, short)


def test_the_command_line_reports_success_and_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    train = stream(300)
    write_sketch(tmp_path / "s", train)
    write_data(tmp_path / "d", train, stream(40, seed=3))
    args = ["--sketch", str(tmp_path / "s"), "--data", str(tmp_path / "d")]
    assert main(args) == 0
    assert "matches the training shard" in capsys.readouterr().out

    sums = np.fromfile(tmp_path / "s" / "sums.f32", dtype="<f4")
    sums[np.flatnonzero(sums)[:50]] += 0.5
    sums.tofile(tmp_path / "s" / "sums.f32")
    assert main([*args, "--sample-rows", "1000"]) == 1
    assert "does not match" in capsys.readouterr().err

    assert main(["--sketch", str(tmp_path / "none"), "--data", str(tmp_path / "d")]) == 1
    assert capsys.readouterr().err.startswith("error: ")
    assert main(["--sketch", str(tmp_path / "s"), "--data", str(tmp_path / "none")]) == 1
    assert capsys.readouterr().err.startswith("error: ")
