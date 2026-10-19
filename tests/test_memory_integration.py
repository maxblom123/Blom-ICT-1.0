import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from blom import GPT, ModelConfig
from blom.checkpoint import CheckpointError, load_checkpoint, save_checkpoint
from blom.config import ConfigError
from blom.data import DataError, Dataset, eval_batches, train_batch, train_starts
from blom.memory import (
    ABSENT,
    NgramHasher,
    SketchError,
    SketchReader,
    eval_batches_with_memory,
    train_batch_with_memory,
)
from blom.optim import build_optimizer
from blom.run import CHECKPOINT_NAME, RUN_NAME, main, run
from blom.sample import main as sample_main
from blom.train import TrainError, evaluate, train_step
from blom.train_config import TrainConfig

CPU = torch.device("cpu")
SEED = 5
VOCAB = 64
PART_DIM = 4

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

CONFIG_TOML = """[model]
vocab_size = 64
d_model = 32
n_layer = 2
n_head = 4
n_kv_head = 2
d_ff = 64
max_seq = 16

[train]
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


def tokens_of(length: int, seed: int, vocab: int = 8) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, vocab, size=length).astype(np.int64)


def make_data(directory: Path, train: np.ndarray, val: np.ndarray, vocab: int = VOCAB) -> Path:
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
    return directory


def build_sketch(
    directory: Path,
    tokens: np.ndarray,
    *,
    rows: int = 1009,
    orders: tuple[int, ...] = (2, 3),
    n_hash: int = 2,
    vocab: int = VOCAB,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    hasher = NgramHasher(SEED, rows, orders, n_hash)
    projection = (
        np.random.default_rng(1).standard_normal((vocab, PART_DIM)) / math.sqrt(PART_DIM)
    ).astype(np.float32)
    padded = np.concatenate([np.full(hasher.history, ABSENT, dtype=np.int64), tokens])
    table_rows = hasher.rows_for(padded[None, :-1])[0]
    sums = np.zeros((rows, PART_DIM), dtype=np.float64)
    counts = np.zeros(rows, dtype=np.int64)
    for lookup in range(hasher.lookup_count):
        np.add.at(sums, table_rows[:, lookup], projection[tokens[1:]].astype(np.float64))
        np.add.at(counts, table_rows[:, lookup], 1)
    projection.astype("<f4").tofile(directory / "projection.f32")
    sums.astype("<f4").tofile(directory / "sums.f32")
    counts.astype("<u4").tofile(directory / "counts.u32")
    manifest = {
        "format": "sketch-v1",
        "seed": SEED,
        "rows": rows,
        "orders": list(orders),
        "hashes": n_hash,
        "part_dim": PART_DIM,
        "vocab_size": vocab,
        "lookups": hasher.lookup_count,
        "pairs": len(tokens) - 1,
        "source_tokens": len(tokens),
        "files": {
            "projection": "projection.f32",
            "sums": "sums.f32",
            "counts": "counts.u32",
        },
    }
    (directory / "sketch.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


class World:
    def __init__(self, root: Path, train_length: int = 3000, val_length: int = 400) -> None:
        self.train = tokens_of(train_length, 11)
        self.val = tokens_of(val_length, 12)
        self.data = make_data(root / "data", self.train, self.val)
        self.sketch = build_sketch(root / "sketch", self.train)
        self.dataset = Dataset.open(self.data)
        self.reader = SketchReader(self.sketch)
        self.config = replace(
            MODEL,
            memory_dim=self.reader.feature_dim,
            memory_scale=math.sqrt(self.reader.part_dim),
        )


class Recorder:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def __call__(self, record: dict[str, object]) -> None:
        self.records.append(record)


def do_run(
    world: World,
    out: Path,
    *,
    sketch: Path | None = None,
    until: int | None = None,
    recorder: Recorder | None = None,
    model: ModelConfig = MODEL,
) -> None:
    run(
        model,
        TRAIN,
        data_dir=world.data,
        out_dir=out,
        device=CPU,
        amp=False,
        log=recorder if recorder is not None else Recorder(),
        until=until,
        sketch_dir=sketch,
    )


def test_memory_fields_are_validated() -> None:
    for change in (
        {"memory_dim": -1},
        {"memory_scale": 0.0},
        {"memory_scale": -1.0},
        {"memory_layer": 2},
        {"memory_layer": -1},
    ):
        with pytest.raises(ConfigError):
            replace(MODEL, **change)
    assert MODEL.memory_dim == 0


def test_memory_parameters_are_counted() -> None:
    config = replace(MODEL, memory_dim=16, memory_scale=2.0)
    model = GPT(config)
    assert sum(p.numel() for p in model.parameters()) == config.param_count
    assert config.param_count - MODEL.param_count == 16 * 32 + 32 + 1


def test_a_model_without_memory_refuses_features() -> None:
    model = GPT(MODEL)
    tokens = torch.zeros(2, 5, dtype=torch.long)
    with pytest.raises(ValueError, match="no memory"):
        model(tokens, mem=torch.zeros(2, 5, 8))


def test_a_memory_model_needs_features_of_the_right_shape() -> None:
    model = GPT(replace(MODEL, memory_dim=8))
    tokens = torch.zeros(2, 5, dtype=torch.long)
    with pytest.raises(ValueError, match="needs memory"):
        model(tokens)
    with pytest.raises(ValueError, match="shape"):
        model(tokens, mem=torch.zeros(2, 4, 8))
    with pytest.raises(ValueError, match="shape"):
        model(tokens, mem=torch.zeros(2, 5, 7))


def test_silent_memory_changes_nothing_and_active_memory_does() -> None:
    torch.manual_seed(0)
    base = GPT(MODEL)
    memory_model = GPT(replace(MODEL, memory_dim=8))
    missing, unexpected = memory_model.load_state_dict(base.state_dict(), strict=False)
    assert unexpected == []
    assert all(name.startswith("memory.") for name in missing)
    tokens = torch.randint(0, VOCAB, (2, 7))
    expected = base(tokens)
    assert torch.equal(memory_model(tokens, mem=torch.zeros(2, 7, 8)), expected)
    assert not torch.equal(memory_model(tokens, mem=torch.randn(2, 7, 8)), expected)


def test_gradients_reach_the_memory_parameters() -> None:
    torch.manual_seed(0)
    model = GPT(replace(MODEL, memory_dim=8))
    tokens = torch.randint(0, VOCAB, (2, 7))
    model(tokens, mem=torch.randn(2, 7, 8)).sum().backward()
    assert model.memory is not None
    for parameter in model.memory.parameters():
        assert parameter.grad is not None
        assert float(parameter.grad.abs().sum()) > 0


def test_the_injection_layer_matters() -> None:
    torch.manual_seed(0)
    first = GPT(replace(MODEL, memory_dim=8, memory_layer=0))
    second = GPT(replace(MODEL, memory_dim=8, memory_layer=1))
    second.load_state_dict(first.state_dict())
    tokens = torch.randint(0, VOCAB, (2, 7))
    mem = torch.randn(2, 7, 8)
    assert not torch.allclose(first(tokens, mem=mem), second(tokens, mem=mem))


def test_incremental_decoding_with_memory_matches_a_full_pass() -> None:
    torch.manual_seed(0)
    model = GPT(replace(MODEL, memory_dim=8)).eval()
    tokens = torch.randint(0, VOCAB, (2, 10))
    mem = torch.randn(2, 10, 8)
    with torch.no_grad():
        full = model(tokens, mem=mem)
        caches = model.new_caches()
        head = model(tokens[:, :6], caches, mem[:, :6])
        tail = model(tokens[:, 6:], caches, mem[:, 6:])
    assert torch.allclose(torch.cat([head, tail], dim=1), full, atol=1e-5)


def test_training_windows_are_the_same_with_and_without_memory(tmp_path: Path) -> None:
    world = World(tmp_path)
    shard = world.dataset.train
    for step in (0, 1, 7):
        plain = train_batch(shard, batch_size=4, seq_len=8, seed=3, step=step)
        inputs, targets, mem = train_batch_with_memory(
            shard, world.reader, batch_size=4, seq_len=8, seed=3, step=step
        )
        assert torch.equal(inputs, plain[0])
        assert torch.equal(targets, plain[1])
        assert mem.shape == (4, 8, world.reader.feature_dim)
        assert mem.dtype == torch.float32


def test_training_starts_pick_the_windows_train_batch_uses(tmp_path: Path) -> None:
    world = World(tmp_path)
    shard = world.dataset.train
    starts = train_starts(shard, batch_size=4, seq_len=8, seed=3, step=2)
    inputs, _ = train_batch(shard, batch_size=4, seq_len=8, seed=3, step=2)
    for row, start in enumerate(starts):
        assert inputs[row].tolist() == world.train[start : start + 8].tolist()


def test_training_features_leave_the_pair_itself_out(tmp_path: Path) -> None:
    world = World(tmp_path)
    reader = world.reader
    starts = train_starts(world.dataset.train, batch_size=4, seq_len=8, seed=3, step=1)
    _, _, mem = train_batch_with_memory(
        world.dataset.train, reader, batch_size=4, seq_len=8, seed=3, step=1
    )
    padded = np.concatenate([np.full(reader.history, ABSENT, dtype=np.int64), world.train])
    for row, start in enumerate(starts):
        context = padded[start : start + reader.history + 8][None, :]
        target = world.train[start + 1 : start + 9][None, :]
        leave_out = reader.features(context, targets=target)
        keep_in = reader.features(context)
        assert np.allclose(mem[row].numpy(), leave_out[0], atol=1e-6)
        assert not np.allclose(leave_out, keep_in)


def test_validation_features_are_plain_lookups(tmp_path: Path) -> None:
    world = World(tmp_path)
    reader = world.reader
    shard = world.dataset.val
    plain = list(eval_batches(shard, batch_size=3, seq_len=8))
    withmem = list(eval_batches_with_memory(shard, reader, batch_size=3, seq_len=8))
    assert len(plain) == len(withmem)
    padded = np.concatenate([np.full(reader.history, ABSENT, dtype=np.int64), world.val])
    for (inputs, targets), (inputs2, targets2, _) in zip(plain, withmem, strict=True):
        assert torch.equal(inputs, inputs2)
        assert torch.equal(targets, targets2)
    first = withmem[0]
    for row in range(first[2].shape[0]):
        start = row * 8
        context = padded[start : start + reader.history + 8][None, :]
        assert np.allclose(first[2][row].numpy(), reader.features(context)[0], atol=1e-6)


def test_a_shard_too_short_for_one_window_is_reported(tmp_path: Path) -> None:
    world = World(tmp_path, val_length=6)
    with pytest.raises(DataError, match="need at least"):
        next(eval_batches_with_memory(world.dataset.val, world.reader, batch_size=2, seq_len=8))


def fresh_memory_model(world: World) -> tuple[GPT, torch.optim.AdamW]:
    torch.manual_seed(0)
    model = GPT(world.config)
    return model, build_optimizer(model, lr=3e-3)


def test_a_training_step_with_memory_is_finite_and_updates_the_memory(tmp_path: Path) -> None:
    world = World(tmp_path)
    model, optimizer = fresh_memory_model(world)
    assert model.memory is not None
    before = model.memory.proj.weight.detach().clone()
    stats = train_step(
        model,
        optimizer,
        world.dataset.train,
        TRAIN,
        step=0,
        device=CPU,
        memory=world.reader,
    )
    assert math.isfinite(stats.loss)
    assert not torch.equal(model.memory.proj.weight, before)


def test_training_with_memory_is_deterministic(tmp_path: Path) -> None:
    world = World(tmp_path)
    losses = []
    for _ in range(2):
        model, optimizer = fresh_memory_model(world)
        run_losses = [
            train_step(
                model,
                optimizer,
                world.dataset.train,
                TRAIN,
                step=step,
                device=CPU,
                memory=world.reader,
            ).loss
            for step in range(3)
        ]
        losses.append(run_losses)
    assert losses[0] == losses[1]


def test_a_memory_model_cannot_be_trained_without_its_sketch(tmp_path: Path) -> None:
    world = World(tmp_path)
    model, optimizer = fresh_memory_model(world)
    with pytest.raises(ValueError, match="needs memory"):
        train_step(model, optimizer, world.dataset.train, TRAIN, step=0, device=CPU)
    with pytest.raises(ValueError, match="needs memory"):
        evaluate(model, world.dataset.val, TRAIN, device=CPU)


def test_evaluation_with_memory_is_finite_and_repeatable(tmp_path: Path) -> None:
    world = World(tmp_path)
    model, _ = fresh_memory_model(world)
    first = evaluate(model, world.dataset.val, TRAIN, device=CPU, memory=world.reader)
    second = evaluate(model, world.dataset.val, TRAIN, device=CPU, memory=world.reader)
    assert math.isfinite(first)
    assert first == second


def test_a_checkpoint_carries_the_sketch_description(tmp_path: Path) -> None:
    world = World(tmp_path)
    model, optimizer = fresh_memory_model(world)
    path = tmp_path / "with.pt"
    save_checkpoint(
        path, model=model, optimizer=optimizer, step=3, memory=world.reader.fingerprint()
    )
    loaded = load_checkpoint(path)
    assert loaded.memory == world.reader.fingerprint()
    assert loaded.config == world.config
    plain_model = GPT(MODEL)
    plain = tmp_path / "plain.pt"
    save_checkpoint(
        plain, model=plain_model, optimizer=build_optimizer(plain_model, lr=1e-3), step=1
    )
    assert load_checkpoint(plain).memory is None


def test_checkpoints_with_unknown_or_malformed_extras_are_refused(tmp_path: Path) -> None:
    model = GPT(MODEL)
    optimizer = build_optimizer(model, lr=1e-3)
    path = tmp_path / "ok.pt"
    save_checkpoint(path, model=model, optimizer=optimizer, step=1)
    payload = torch.load(path, weights_only=True)
    torch.save({**payload, "surprise": 1}, tmp_path / "extra.pt")
    with pytest.raises(CheckpointError, match="not a checkpoint"):
        load_checkpoint(tmp_path / "extra.pt")
    torch.save({**payload, "memory": [1, 2]}, tmp_path / "bad.pt")
    with pytest.raises(CheckpointError, match="memory description"):
        load_checkpoint(tmp_path / "bad.pt")
    missing = {key: value for key, value in payload.items() if key != "optimizer"}
    torch.save(missing, tmp_path / "short.pt")
    with pytest.raises(CheckpointError, match="not a checkpoint"):
        load_checkpoint(tmp_path / "short.pt")


def test_a_run_with_a_sketch_records_it_everywhere(tmp_path: Path) -> None:
    world = World(tmp_path)
    recorder = Recorder()
    do_run(world, tmp_path / "out", sketch=world.sketch, recorder=recorder)
    start = recorder.records[0]
    assert start["memory"] == world.reader.fingerprint()
    assert start["parameters"] == world.config.param_count
    stored = json.loads((tmp_path / "out" / RUN_NAME).read_text(encoding="utf-8"))
    assert stored["memory"] == world.reader.fingerprint()
    assert stored["model"]["memory_dim"] == world.reader.feature_dim
    checkpoint = load_checkpoint(tmp_path / "out" / CHECKPOINT_NAME)
    assert checkpoint.memory == world.reader.fingerprint()
    assert checkpoint.config == world.config
    assert checkpoint.step == 8


def test_a_stopped_and_resumed_run_with_memory_matches_an_uninterrupted_one(
    tmp_path: Path,
) -> None:
    world = World(tmp_path)
    do_run(world, tmp_path / "straight", sketch=world.sketch)
    do_run(world, tmp_path / "resumed", sketch=world.sketch, until=3)
    recorder = Recorder()
    do_run(world, tmp_path / "resumed", sketch=world.sketch, recorder=recorder)
    assert {"event": "resume", "step": 3} in recorder.records
    straight = load_checkpoint(tmp_path / "straight" / CHECKPOINT_NAME)
    resumed = load_checkpoint(tmp_path / "resumed" / CHECKPOINT_NAME)
    for name, tensor in straight.model.items():
        assert torch.equal(resumed.model[name], tensor), name


def test_a_run_cannot_be_resumed_with_another_or_no_sketch(tmp_path: Path) -> None:
    world = World(tmp_path)
    do_run(world, tmp_path / "out", sketch=world.sketch, until=3)
    other = build_sketch(tmp_path / "other", world.train, rows=2003)
    with pytest.raises(TrainError, match="different configuration"):
        do_run(world, tmp_path / "out", sketch=other)
    with pytest.raises(TrainError, match="different configuration"):
        do_run(world, tmp_path / "out")


def test_a_checkpoint_from_another_sketch_is_refused_even_if_the_run_file_is_edited(
    tmp_path: Path,
) -> None:
    world = World(tmp_path)
    do_run(world, tmp_path / "out", sketch=world.sketch, until=3)
    other = build_sketch(tmp_path / "other", world.train, rows=2003)
    run_file = tmp_path / "out" / RUN_NAME
    stored = json.loads(run_file.read_text(encoding="utf-8"))
    stored["memory"] = SketchReader(other).fingerprint()
    run_file.write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(TrainError, match="different sketch"):
        do_run(world, tmp_path / "out", sketch=other)


def test_a_sketch_from_other_data_is_refused_before_anything_is_written(tmp_path: Path) -> None:
    world = World(tmp_path)
    other = build_sketch(tmp_path / "other", world.train[:2000])
    with pytest.raises(SketchError, match="training tokens"):
        do_run(world, tmp_path / "out", sketch=other)
    assert not (tmp_path / "out").exists()


def test_a_model_that_wants_memory_needs_a_sketch(tmp_path: Path) -> None:
    world = World(tmp_path)
    with pytest.raises(TrainError, match="no --sketch"):
        do_run(world, tmp_path / "out", model=world.config)


def cli_files(tmp_path: Path, world: World) -> list[str]:
    config = tmp_path / "config.toml"
    config.write_text(CONFIG_TOML, encoding="utf-8")
    return [
        "--config",
        str(config),
        "--data",
        str(world.data),
        "--out",
        str(tmp_path / "cli"),
        "--device",
        "cpu",
    ]


def test_the_command_line_trains_with_a_sketch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    world = World(tmp_path)
    assert main([*cli_files(tmp_path, world), "--sketch", str(world.sketch)]) == 0
    first = json.loads(capsys.readouterr().out.splitlines()[0])
    assert first["memory"]["rows"] == 1009
    assert load_checkpoint(tmp_path / "cli" / CHECKPOINT_NAME).memory is not None


def test_the_command_line_reports_a_bad_sketch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    world = World(tmp_path)
    assert main([*cli_files(tmp_path, world), "--sketch", str(tmp_path / "nothing")]) == 1
    assert capsys.readouterr().err.startswith("error: ")


def make_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=300,
        special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(["hello world, once upon a time. " * 40], trainer)
    return tokenizer


def test_sampling_from_a_memory_checkpoint_is_refused_clearly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tokenizer = make_tokenizer()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    config = replace(MODEL, vocab_size=tokenizer.get_vocab_size(), memory_dim=8, max_seq=64)
    model = GPT(config)
    save_checkpoint(
        tmp_path / "ckpt.pt", model=model, optimizer=build_optimizer(model, lr=1e-3), step=1
    )
    code = sample_main(
        [
            "--checkpoint",
            str(tmp_path / "ckpt.pt"),
            "--tokenizer",
            str(tmp_path / "tokenizer.json"),
            "--device",
            "cpu",
        ]
    )
    assert code == 1
    assert "sketch memory" in capsys.readouterr().err
