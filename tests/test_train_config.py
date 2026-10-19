from dataclasses import replace
from pathlib import Path

import pytest

from blom.train_config import TrainConfig, TrainConfigError

BASE = {
    "batch_size": 4,
    "seq_len": 8,
    "peak_lr": 1e-3,
    "min_lr": 1e-4,
    "warmup_steps": 10,
    "decay_steps": 20,
    "total_steps": 100,
}


def test_defaults_fill_the_optional_fields() -> None:
    config = TrainConfig.from_dict(BASE)
    assert config.grad_accum == 1
    assert config.seed == 0
    assert config.weight_decay == 0.1
    assert config.max_grad_norm == 1.0


def test_tokens_per_step_counts_the_whole_accumulated_batch() -> None:
    config = replace(TrainConfig.from_dict(BASE), grad_accum=3)
    assert config.tokens_per_step == 4 * 3 * 8


def test_schedule_is_built_from_the_config() -> None:
    schedule = TrainConfig.from_dict(BASE).schedule
    assert schedule.peak_lr == 1e-3
    assert schedule.total_steps == 100
    assert schedule.lr_at(9) == pytest.approx(1e-3)


def test_integers_are_accepted_for_float_fields() -> None:
    config = TrainConfig.from_dict({**BASE, "min_lr": 0})
    assert config.schedule.lr_at(99) == 0


def test_load_reads_the_train_table(tmp_path: Path) -> None:
    path = tmp_path / "run.toml"
    path.write_text(
        "[model]\nd_model = 8\n\n[train]\nbatch_size = 2\nseq_len = 16\npeak_lr = 3e-4\n"
        "min_lr = 3e-5\nwarmup_steps = 5\ndecay_steps = 5\ntotal_steps = 50\ngrad_accum = 4\n",
        encoding="utf-8",
    )
    config = TrainConfig.load(path)
    assert (config.batch_size, config.seq_len, config.grad_accum) == (2, 16, 4)
    assert config.peak_lr == 3e-4


def test_load_rejects_a_missing_table_file_and_bad_toml(tmp_path: Path) -> None:
    path = tmp_path / "run.toml"
    path.write_text("[model]\nd_model = 8\n", encoding="utf-8")
    with pytest.raises(TrainConfigError, match=r"no \[train\] table"):
        TrainConfig.load(path)
    with pytest.raises(TrainConfigError, match="cannot read"):
        TrainConfig.load(tmp_path / "missing.toml")
    path.write_text("[train\n", encoding="utf-8")
    with pytest.raises(TrainConfigError, match="cannot read"):
        TrainConfig.load(path)


def test_unknown_and_missing_keys_are_rejected() -> None:
    with pytest.raises(TrainConfigError, match="unknown train config keys: surprise"):
        TrainConfig.from_dict({**BASE, "surprise": 1})
    with pytest.raises(TrainConfigError):
        TrainConfig.from_dict({"batch_size": 4})


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("batch_size", True, "integer"),
        ("batch_size", 2.0, "integer"),
        ("seq_len", "8", "integer"),
        ("peak_lr", "fast", "number"),
        ("peak_lr", True, "number"),
        ("peak_lr", float("nan"), "finite"),
        ("min_lr", float("inf"), "finite"),
        ("batch_size", 0, "positive"),
        ("grad_accum", 0, "positive"),
        ("seq_len", -1, "positive"),
        ("eval_every", 0, "positive"),
        ("eval_batches", 0, "positive"),
        ("checkpoint_every", 0, "positive"),
        ("log_every", 0, "positive"),
        ("max_grad_norm", 0.0, "positive"),
        ("seed", -1, "negative"),
        ("weight_decay", -0.1, "negative"),
        ("warmup_steps", 0, "positive"),
        ("decay_steps", 0, "positive"),
        ("warmup_steps", 90, "must not exceed"),
        ("min_lr", 1.0, "min_lr"),
    ],
)
def test_invalid_values_are_rejected(key: str, value: object, message: str) -> None:
    with pytest.raises(TrainConfigError, match=message):
        TrainConfig.from_dict({**BASE, key: value})
