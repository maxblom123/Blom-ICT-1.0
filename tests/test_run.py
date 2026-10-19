import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

import blom.run as run_module
from blom import ModelConfig
from blom.checkpoint import load_checkpoint
from blom.data import DataError
from blom.run import (
    CHECKPOINT_NAME,
    METRICS_NAME,
    RUN_NAME,
    RunResult,
    main,
    resolve_device,
    run,
)
from blom.train import TrainError
from blom.train_config import TrainConfig

CPU = torch.device("cpu")

MODEL = ModelConfig(
    vocab_size=64, d_model=32, n_layer=2, n_head=4, n_kv_head=2, d_ff=64, max_seq=16
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


def make_data(directory: Path, vocab: int = 64) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, length in (("train", 3000), ("val", 400)):
        tokens = (np.arange(length) % 8).astype("<u2")
        tokens.tofile(directory / f"{name}.bin")
        counts[name] = {"tokens": length}
    meta = {"format": "u16-le", "vocab_size": vocab, "end_of_text_id": 0, **counts}
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return directory


class Recorder:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def __call__(self, record: dict[str, object]) -> None:
        self.records.append(record)

    def steps(self, event: str) -> list[object]:
        return [r["step"] for r in self.records if r["event"] == event]


def do_run(
    tmp_path: Path,
    *,
    out: str = "out",
    train: TrainConfig = TRAIN,
    until: int | None = None,
    recorder: Recorder | None = None,
    vocab: int = 64,
) -> RunResult:
    return run(
        MODEL,
        train,
        data_dir=make_data(tmp_path / "data", vocab),
        out_dir=tmp_path / out,
        device=CPU,
        amp=False,
        log=recorder if recorder is not None else Recorder(),
        until=until,
    )


def test_a_full_run_trains_evaluates_and_checkpoints(tmp_path: Path) -> None:
    recorder = Recorder()
    result = do_run(tmp_path, recorder=recorder)
    assert result.step == 8
    assert result.train_loss is not None
    assert result.val_loss is not None
    assert recorder.steps("train") == [2, 4, 6, 8]
    assert recorder.steps("eval") == [4, 8]
    assert recorder.steps("checkpoint") == [4, 8]
    start = recorder.records[0]
    assert start["event"] == "start"
    assert start["parameters"] == MODEL.param_count
    assert start["tokens_per_step"] == 32
    assert load_checkpoint(tmp_path / "out" / CHECKPOINT_NAME).step == 8
    assert (tmp_path / "out" / RUN_NAME).is_file()


def test_stopping_early_still_logs_evaluates_and_checkpoints_the_last_step(
    tmp_path: Path,
) -> None:
    recorder = Recorder()
    result = do_run(tmp_path, until=3, recorder=recorder)
    assert result.step == 3
    assert recorder.steps("train") == [2, 3]
    assert recorder.steps("eval") == [3]
    assert recorder.steps("checkpoint") == [3]
    assert load_checkpoint(tmp_path / "out" / CHECKPOINT_NAME).step == 3


def test_the_run_learns_the_pattern(tmp_path: Path) -> None:
    train = replace(TRAIN, total_steps=40, warmup_steps=4, decay_steps=10, eval_every=20)
    train = replace(train, checkpoint_every=20, log_every=10)
    recorder = Recorder()
    do_run(tmp_path, train=train, recorder=recorder)
    losses = [r["loss"] for r in recorder.records if r["event"] == "train"]
    assert isinstance(losses[0], float)
    assert isinstance(losses[-1], float)
    assert losses[0] > 3.0
    assert losses[-1] < 1.0


def test_a_stopped_and_resumed_run_matches_an_uninterrupted_one(tmp_path: Path) -> None:
    do_run(tmp_path, out="straight")
    recorder = Recorder()
    do_run(tmp_path, out="resumed", until=3)
    do_run(tmp_path, out="resumed", recorder=recorder)
    assert {"event": "resume", "step": 3} in recorder.records
    straight = load_checkpoint(tmp_path / "straight" / CHECKPOINT_NAME)
    resumed = load_checkpoint(tmp_path / "resumed" / CHECKPOINT_NAME)
    assert straight.step == resumed.step == 8
    for name, tensor in straight.model.items():
        assert torch.equal(resumed.model[name], tensor), name


def test_running_a_finished_run_again_does_nothing(tmp_path: Path) -> None:
    do_run(tmp_path)
    before = (tmp_path / "out" / CHECKPOINT_NAME).read_bytes()
    recorder = Recorder()
    result = do_run(tmp_path, recorder=recorder)
    assert result.step == 8
    assert result.train_loss is None
    assert recorder.steps("train") == []
    assert recorder.records[-1]["event"] == "nothing_to_do"
    assert (tmp_path / "out" / CHECKPOINT_NAME).read_bytes() == before


def test_a_changed_configuration_is_refused_in_an_existing_run_directory(tmp_path: Path) -> None:
    do_run(tmp_path, until=2)
    with pytest.raises(TrainError, match="different configuration"):
        do_run(tmp_path, train=replace(TRAIN, peak_lr=1e-3))


def test_a_corrupt_run_description_is_reported(tmp_path: Path) -> None:
    do_run(tmp_path, until=2)
    (tmp_path / "out" / RUN_NAME).write_text("{", encoding="utf-8")
    with pytest.raises(TrainError, match="cannot read"):
        do_run(tmp_path)


@pytest.mark.parametrize("until", [0, -1, 9])
def test_until_outside_the_run_is_rejected(tmp_path: Path, until: int) -> None:
    with pytest.raises(TrainError, match="until must be"):
        do_run(tmp_path, until=until)


def test_a_sequence_longer_than_the_model_supports_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(TrainError, match="max_seq"):
        do_run(tmp_path, train=replace(TRAIN, seq_len=32))


def test_a_vocabulary_mismatch_is_rejected_before_anything_is_written(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="vocab"):
        do_run(tmp_path, vocab=128)
    assert not (tmp_path / "out").exists()


def test_a_non_finite_validation_loss_stops_the_run_without_a_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_module, "evaluate", lambda *_a, **_k: float("nan"))
    with pytest.raises(TrainError, match="validation loss is nan at step 4"):
        do_run(tmp_path)
    assert not (tmp_path / "out" / CHECKPOINT_NAME).exists()


def test_cuda_is_refused_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == torch.device("cpu")
    assert resolve_device("cpu") == torch.device("cpu")
    with pytest.raises(TrainError, match="no CUDA device"):
        resolve_device("cuda")


def test_auto_picks_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == torch.device("cuda")
    assert resolve_device("cuda") == torch.device("cuda")


def cli_files(tmp_path: Path) -> list[str]:
    config = tmp_path / "run.toml"
    config.write_text(CONFIG_TOML, encoding="utf-8")
    data = make_data(tmp_path / "data")
    return ["--config", str(config), "--data", str(data), "--out", str(tmp_path / "out")]


def test_the_command_line_runs_and_writes_json_lines(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main([*cli_files(tmp_path), "--device", "cpu"])
    assert code == 0
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    written = [
        json.loads(line)
        for line in (tmp_path / "out" / METRICS_NAME).read_text(encoding="utf-8").splitlines()
    ]
    assert printed == written
    assert printed[0]["event"] == "start"
    assert printed[0]["amp"] is False
    assert [r["step"] for r in printed if r["event"] == "eval"] == [4, 8]


def test_the_command_line_resumes_and_appends_to_the_metrics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    files = cli_files(tmp_path)
    assert main([*files, "--device", "cpu", "--until", "4"]) == 0
    assert main([*files, "--device", "cpu"]) == 0
    capsys.readouterr()
    lines = (tmp_path / "out" / METRICS_NAME).read_text(encoding="utf-8").splitlines()
    events = [json.loads(line)["event"] for line in lines]
    assert events.count("start") == 2
    assert events.count("resume") == 1
    assert load_checkpoint(tmp_path / "out" / CHECKPOINT_NAME).step == 8


@pytest.mark.filterwarnings("ignore:Mismatch dtype:UserWarning")
def test_the_amp_flag_is_honoured_even_on_cpu(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([*cli_files(tmp_path), "--device", "cpu", "--amp", "--until", "1"]) == 0
    first = json.loads(capsys.readouterr().out.splitlines()[0])
    assert first["amp"] is True


def test_the_command_line_reports_errors_with_exit_code_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "missing.toml"
    args = ["--config", str(missing), "--data", str(tmp_path), "--out", str(tmp_path / "out")]
    assert main([*args, "--device", "cpu"]) == 1
    assert capsys.readouterr().err.startswith("error: ")


def test_the_command_line_reports_a_bad_until(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([*cli_files(tmp_path), "--device", "cpu", "--until", "99"]) == 1
    assert "until must be" in capsys.readouterr().err


def test_an_unknown_device_is_a_usage_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([*cli_files(tmp_path), "--device", "tpu"])
    assert excinfo.value.code == 2
