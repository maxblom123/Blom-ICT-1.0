import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from blom import GPT, IGNORE_INDEX, ModelConfig
from blom.checkpoint import load_checkpoint, save_checkpoint
from blom.data import DataError, Shard
from blom.optim import build_optimizer
from blom.run import CHECKPOINT_NAME
from blom.sft import (
    ChatShard,
    eval_batches,
    main,
    open_chat_data,
    sft_evaluate,
    sft_run,
    sft_step,
    train_batch,
)
from blom.train import TrainError
from blom.train_config import TrainConfig

CPU = torch.device("cpu")
VOCAB = 32

MODEL = ModelConfig(
    vocab_size=VOCAB, d_model=32, n_layer=2, n_head=4, n_kv_head=2, d_ff=64, max_seq=16
)

TRAIN = TrainConfig(
    batch_size=4,
    seq_len=8,
    peak_lr=3e-3,
    min_lr=3e-4,
    warmup_steps=2,
    decay_steps=4,
    total_steps=8,
    log_every=2,
    eval_every=4,
    eval_batches=2,
    checkpoint_every=4,
)


def conversations(count: int, seed: int) -> list[tuple[list[int], list[int]]]:
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(count):
        length = int(rng.integers(6, 14))
        tokens = rng.integers(1, VOCAB, size=length).tolist()
        learn = [0] * (length // 3) + [1] * (length - length // 3)
        result.append(([*tokens, 0], [*learn, 0]))
    return result


def write_pair(
    directory: Path, name: str, convs: list[tuple[list[int], list[int]]]
) -> tuple[Path, Path]:
    tokens = [token for conv, _ in convs for token in conv]
    mask = [flag for _, learn in convs for flag in learn]
    bin_path = directory / f"{name}.bin"
    mask_path = directory / f"{name}.mask"
    np.asarray(tokens, dtype="<u2").tofile(bin_path)
    np.asarray(mask, dtype="u1").tofile(mask_path)
    return bin_path, mask_path


def make_chat_data(directory: Path, train_count: int = 60, val_count: int = 12) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    meta: dict[str, object] = {
        "format": "u16-le",
        "kind": "chat",
        "mask_format": "u8",
        "vocab_size": VOCAB,
        "end_of_text_id": 0,
    }
    for name, count, seed in (("train", train_count, 1), ("val", val_count, 2)):
        convs = conversations(count, seed)
        write_pair(directory, name, convs)
        meta[name] = {"documents": count, "tokens": sum(len(c) for c, _ in convs)}
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return directory


def chat_shard(
    tmp_path: Path, count: int = 30
) -> tuple[ChatShard, list[tuple[list[int], list[int]]]]:
    convs = conversations(count, 5)
    bin_path, mask_path = write_pair(tmp_path, "x", convs)
    return ChatShard(Shard(bin_path), mask_path, 0), convs


def save_pretrained(path: Path, seed: int = 0) -> None:
    torch.manual_seed(seed)
    model = GPT(MODEL)
    save_checkpoint(path, model=model, optimizer=build_optimizer(model, lr=1e-3), step=100)


def test_starts_are_the_first_token_of_each_conversation(tmp_path: Path) -> None:
    shard, convs = chat_shard(tmp_path)
    expected = np.cumsum([0] + [len(c) for c, _ in convs])[:-1]
    assert shard.starts.tolist() == expected.tolist()


def test_starts_are_cached_and_the_cache_is_checked(tmp_path: Path) -> None:
    shard, _ = chat_shard(tmp_path)
    first = shard.starts.copy()
    cache = tmp_path / "x.starts.npy"
    assert cache.is_file()
    again = ChatShard(Shard(tmp_path / "x.bin"), tmp_path / "x.mask", 0)
    assert again.starts.tolist() == first.tolist()
    np.save(cache, np.array([0, 3, 5]))
    broken = ChatShard(Shard(tmp_path / "x.bin"), tmp_path / "x.mask", 0)
    assert broken.starts.tolist() == first.tolist()


def test_a_stale_cache_is_rebuilt(tmp_path: Path) -> None:
    shard, convs = chat_shard(tmp_path)
    cache = tmp_path / "x.starts.npy"
    shard.starts  # noqa: B018
    older = cache.stat().st_mtime - 100
    import os

    os.utime(cache, (older, older))
    fresh = ChatShard(Shard(tmp_path / "x.bin"), tmp_path / "x.mask", 0)
    assert len(fresh.starts) == len(convs)
    assert cache.stat().st_mtime > older


def test_mask_must_match_the_shard_length(tmp_path: Path) -> None:
    bin_path, mask_path = write_pair(tmp_path, "x", conversations(5, 1))
    mask_path.write_bytes(mask_path.read_bytes()[:-1])
    with pytest.raises(DataError, match="expected"):
        ChatShard(Shard(bin_path), mask_path, 0)
    with pytest.raises(DataError, match="cannot read mask"):
        ChatShard(Shard(bin_path), tmp_path / "missing.mask", 0)


def test_window_targets_are_ignored_where_the_mask_is_zero(tmp_path: Path) -> None:
    shard, convs = chat_shard(tmp_path)
    tokens = [token for conv, _ in convs for token in conv]
    mask = [flag for _, learn in convs for flag in learn]
    for start in (0, 3, 17):
        inputs, targets = shard.window(start, 8)
        assert inputs.tolist() == tokens[start : start + 8]
        for offset in range(8):
            position = start + offset + 1
            expected = tokens[position] if mask[position] else IGNORE_INDEX
            assert targets[offset] == expected


def test_train_batches_start_at_conversation_boundaries_and_are_deterministic(
    tmp_path: Path,
) -> None:
    shard, _ = chat_shard(tmp_path)
    inputs, targets = train_batch(shard, batch_size=6, seq_len=8, seed=3, step=7)
    again, again_targets = train_batch(shard, batch_size=6, seq_len=8, seed=3, step=7)
    other, _ = train_batch(shard, batch_size=6, seq_len=8, seed=3, step=8)
    assert torch.equal(inputs, again)
    assert torch.equal(targets, again_targets)
    assert not torch.equal(inputs, other)
    assert inputs.shape == (6, 8)
    assert inputs.dtype == torch.int64
    all_tokens = np.fromfile(tmp_path / "x.bin", dtype="<u2")
    starts = shard.starts
    for row in inputs.tolist():
        found = [
            int(s)
            for s in starts
            if s + 8 <= len(all_tokens) and all_tokens[s : s + 8].tolist() == row
        ]
        assert found
    assert bool((targets != IGNORE_INDEX).any(dim=1).all())


def test_windows_never_run_past_the_end_of_the_shard(tmp_path: Path) -> None:
    shard, _ = chat_shard(tmp_path, count=4)
    usable = shard.usable_starts(8)
    assert all(int(s) + 9 <= len(shard) for s in usable)
    for step in range(20):
        train_batch(shard, batch_size=2, seq_len=8, seed=0, step=step)


def test_too_short_data_is_an_error(tmp_path: Path) -> None:
    shard, _ = chat_shard(tmp_path, count=1)
    with pytest.raises(DataError, match="no conversation leaves room"):
        shard.usable_starts(40)


def test_windows_without_assistant_text_are_never_used(tmp_path: Path) -> None:
    convs = [
        ([5, 6, 7, 8, 9, 10, 11, 12, 0], [0] * 9),
        ([1, 2, 3, 4, 5, 6, 7, 8, 0], [0, 0, 0, 1, 1, 1, 1, 1, 0]),
    ]
    bin_path, mask_path = write_pair(tmp_path, "y", convs * 6)
    shard = ChatShard(Shard(bin_path), mask_path, 0)
    inputs, _ = train_batch(shard, batch_size=4, seq_len=6, seed=1, step=0)
    assert all(row[0] == 1 for row in inputs.tolist())


def test_all_masked_data_is_an_error(tmp_path: Path) -> None:
    convs = [([5, 6, 7, 8, 9, 10, 11, 12, 0], [0] * 9)] * 6
    bin_path, mask_path = write_pair(tmp_path, "z", convs)
    shard = ChatShard(Shard(bin_path), mask_path, 0)
    with pytest.raises(DataError, match="assistant text"):
        train_batch(shard, batch_size=2, seq_len=6, seed=0, step=0)


def test_eval_batches_are_spread_over_the_data_and_repeatable(tmp_path: Path) -> None:
    shard, _ = chat_shard(tmp_path)
    first = list(eval_batches(shard, batch_size=4, seq_len=8, windows=10))
    again = list(eval_batches(shard, batch_size=4, seq_len=8, windows=10))
    assert sum(inputs.shape[0] for inputs, _ in first) == 10
    assert all(torch.equal(a[0], b[0]) for a, b in zip(first, again, strict=True))
    firsts = {tuple(row) for inputs, _ in first for row in inputs.tolist()}
    assert len(firsts) == 10


def test_open_chat_data_checks_the_kind_and_vocab(tmp_path: Path) -> None:
    data = make_chat_data(tmp_path / "data")
    train, val = open_chat_data(data, vocab_size=VOCAB)
    assert len(train) > len(val) > 0
    with pytest.raises(DataError, match="vocab"):
        open_chat_data(data, vocab_size=VOCAB + 1)
    meta = json.loads((data / "meta.json").read_text(encoding="utf-8"))
    del meta["kind"]
    (data / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(DataError, match="tokenize-chat"):
        open_chat_data(data, vocab_size=VOCAB)


def manual_loss(model: GPT, batches: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
    total = 0.0
    count = 0
    with torch.no_grad():
        for inputs, targets in batches:
            logits = model(inputs)
            total += float(
                F.cross_entropy(
                    logits.reshape(-1, VOCAB),
                    targets.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                    reduction="sum",
                )
            )
            count += int((targets != IGNORE_INDEX).sum())
    return total / count


@pytest.mark.parametrize("accum", [1, 3])
def test_step_loss_is_the_mean_over_every_trained_token(tmp_path: Path, accum: int) -> None:
    shard, _ = chat_shard(tmp_path)
    torch.manual_seed(0)
    model = GPT(MODEL)
    optimizer = build_optimizer(model, lr=1e-3)
    config = replace(TRAIN, grad_accum=accum)
    batches = [
        train_batch(shard, batch_size=4, seq_len=8, seed=0, step=5 * accum + micro)
        for micro in range(accum)
    ]
    expected = manual_loss(model, batches)
    stats = sft_step(model, optimizer, shard, config, step=5, device=CPU)
    assert stats.loss == pytest.approx(expected, rel=1e-4)
    assert stats.trained_tokens == sum(int((t != IGNORE_INDEX).sum()) for _, t in batches)
    assert stats.tokens == config.tokens_per_step


def test_evaluate_scores_only_trained_tokens(tmp_path: Path) -> None:
    shard, _ = chat_shard(tmp_path)
    torch.manual_seed(0)
    model = GPT(MODEL)
    batches = list(eval_batches(shard, batch_size=4, seq_len=8, windows=8))
    value = sft_evaluate(model, shard, replace(TRAIN, eval_batches=2), device=CPU)
    assert value == pytest.approx(manual_loss(model, batches), rel=1e-4)


def test_a_step_changes_the_weights(tmp_path: Path) -> None:
    shard, _ = chat_shard(tmp_path)
    torch.manual_seed(0)
    model = GPT(MODEL)
    before = model.lm_head.weight.detach().clone()
    sft_step(model, build_optimizer(model, lr=1e-3), shard, TRAIN, step=0, device=CPU)
    assert not torch.equal(before, model.lm_head.weight)


def run_sft(
    tmp_path: Path, out: str, *, until: int | None = None, config: TrainConfig = TRAIN
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    sft_run(
        config,
        init=tmp_path / "pre.pt",
        data_dir=tmp_path / "data",
        out_dir=tmp_path / out,
        device=CPU,
        amp=False,
        log=records.append,
        until=until,
    )
    return records


def prepare(tmp_path: Path) -> None:
    make_chat_data(tmp_path / "data")
    save_pretrained(tmp_path / "pre.pt")


def test_a_run_starts_from_the_pretrained_weights_and_learns(tmp_path: Path) -> None:
    prepare(tmp_path)
    config = replace(TRAIN, total_steps=40, decay_steps=10, eval_every=40, checkpoint_every=40)
    records = run_sft(tmp_path, "run", config=config)
    evals = [r for r in records if r["event"] == "eval"]
    assert [r["step"] for r in evals] == [0, 40]
    assert float(str(evals[1]["val_loss"])) < float(str(evals[0]["val_loss"]))
    checkpoint = load_checkpoint(tmp_path / "run" / CHECKPOINT_NAME)
    assert checkpoint.step == 40
    start = next(r for r in records if r["event"] == "start")
    assert start["init"] == str(tmp_path / "pre.pt")


def test_resuming_gives_the_same_weights_as_one_straight_run(tmp_path: Path) -> None:
    prepare(tmp_path)
    run_sft(tmp_path, "straight")
    run_sft(tmp_path, "split", until=4)
    records = run_sft(tmp_path, "split")
    assert any(r["event"] == "resume" and r["step"] == 4 for r in records)
    straight = load_checkpoint(tmp_path / "straight" / CHECKPOINT_NAME)
    split = load_checkpoint(tmp_path / "split" / CHECKPOINT_NAME)
    assert split.step == straight.step == 8
    for name, tensor in straight.model.items():
        assert torch.equal(tensor, split.model[name]), name


def test_a_finished_run_has_nothing_to_do(tmp_path: Path) -> None:
    prepare(tmp_path)
    run_sft(tmp_path, "run")
    records = run_sft(tmp_path, "run")
    assert records[-1]["event"] == "nothing_to_do"


def test_a_new_run_needs_an_init_checkpoint(tmp_path: Path) -> None:
    prepare(tmp_path)
    with pytest.raises(TrainError, match="--init"):
        sft_run(
            TRAIN,
            init=None,
            data_dir=tmp_path / "data",
            out_dir=tmp_path / "run",
            device=CPU,
            amp=False,
            log=lambda _: None,
        )


def test_seq_len_beyond_the_model_is_refused(tmp_path: Path) -> None:
    prepare(tmp_path)
    with pytest.raises(TrainError, match="max_seq"):
        run_sft(tmp_path, "run", config=replace(TRAIN, seq_len=32))


def test_a_changed_config_is_refused_on_resume(tmp_path: Path) -> None:
    prepare(tmp_path)
    run_sft(tmp_path, "run", until=4)
    with pytest.raises(TrainError, match="different configuration"):
        run_sft(tmp_path, "run", config=replace(TRAIN, peak_lr=1e-3))


def test_memory_models_are_refused(tmp_path: Path) -> None:
    make_chat_data(tmp_path / "data")
    config = replace(MODEL, memory_dim=8)
    torch.manual_seed(0)
    model = GPT(config)
    save_checkpoint(
        tmp_path / "pre.pt", model=model, optimizer=build_optimizer(model, lr=1e-3), step=1
    )
    with pytest.raises(TrainError, match="memory"):
        run_sft(tmp_path, "run")


def test_vocab_mismatch_is_refused(tmp_path: Path) -> None:
    make_chat_data(tmp_path / "data")
    config = replace(MODEL, vocab_size=VOCAB + 8)
    torch.manual_seed(0)
    model = GPT(config)
    save_checkpoint(
        tmp_path / "pre.pt", model=model, optimizer=build_optimizer(model, lr=1e-3), step=1
    )
    with pytest.raises(DataError, match="vocab"):
        run_sft(tmp_path, "run")


TOML = """[train]
batch_size = 4
seq_len = 8
peak_lr = 3e-3
min_lr = 3e-4
warmup_steps = 2
decay_steps = 4
total_steps = 8
log_every = 2
eval_every = 4
eval_batches = 2
checkpoint_every = 4
"""


def test_command_line_runs_and_reports_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prepare(tmp_path)
    (tmp_path / "sft.toml").write_text(TOML, encoding="utf-8")
    base = [
        "--config",
        str(tmp_path / "sft.toml"),
        "--data",
        str(tmp_path / "data"),
        "--device",
        "cpu",
    ]
    args = [*base, "--out", str(tmp_path / "run")]
    assert main([*args, "--init", str(tmp_path / "pre.pt")]) == 0
    assert (tmp_path / "run" / "metrics.jsonl").is_file()
    assert load_checkpoint(tmp_path / "run" / CHECKPOINT_NAME).step == 8
    capsys.readouterr()
    assert main([*base, "--out", str(tmp_path / "other")]) == 1
    assert "--init" in capsys.readouterr().err