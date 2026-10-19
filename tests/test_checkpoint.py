from dataclasses import replace
from pathlib import Path

import pytest
import torch

from blom import GPT, ModelConfig, next_token_loss
from blom.checkpoint import (
    FORMAT,
    CheckpointError,
    load_checkpoint,
    restore,
    save_checkpoint,
)
from blom.optim import build_optimizer

CONFIG = ModelConfig(
    vocab_size=64, d_model=32, n_layer=2, n_head=4, n_kv_head=2, d_ff=64, max_seq=16
)


def batch(step: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(step)
    tokens = torch.randint(0, 64, (4, 9), generator=generator)
    return tokens[:, :-1], tokens[:, 1:]


def train(model: GPT, optimizer: torch.optim.AdamW, first: int, last: int) -> None:
    for step in range(first, last):
        inputs, targets = batch(step)
        loss = next_token_loss(model(inputs), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def fresh(seed: int = 0, config: ModelConfig = CONFIG) -> tuple[GPT, torch.optim.AdamW]:
    torch.manual_seed(seed)
    model = GPT(config)
    return model, build_optimizer(model, lr=1e-2)


def test_round_trip_restores_weights_optimizer_and_step(tmp_path: Path) -> None:
    model, optimizer = fresh()
    train(model, optimizer, 0, 3)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model=model, optimizer=optimizer, step=3)
    other, other_optimizer = fresh(seed=99)
    step = restore(load_checkpoint(path), model=other, optimizer=other_optimizer)
    assert step == 3
    for name, tensor in model.state_dict().items():
        assert torch.equal(other.state_dict()[name], tensor), name
    assert other_optimizer.state_dict()["state"].keys() == optimizer.state_dict()["state"].keys()


def test_resumed_training_matches_uninterrupted_training_exactly(tmp_path: Path) -> None:
    straight, straight_optimizer = fresh()
    train(straight, straight_optimizer, 0, 6)

    first, first_optimizer = fresh()
    train(first, first_optimizer, 0, 3)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model=first, optimizer=first_optimizer, step=3)

    resumed, resumed_optimizer = fresh(seed=123)
    step = restore(load_checkpoint(path), model=resumed, optimizer=resumed_optimizer)
    train(resumed, resumed_optimizer, step, 6)

    for name, tensor in straight.state_dict().items():
        assert torch.equal(resumed.state_dict()[name], tensor), name


def test_tied_weights_stay_tied_after_loading(tmp_path: Path) -> None:
    model, optimizer = fresh()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model=model, optimizer=optimizer, step=0)
    other, _ = fresh(seed=5)
    restore(load_checkpoint(path), model=other)
    assert other.lm_head.weight is other.tok.weight


def test_untied_models_round_trip(tmp_path: Path) -> None:
    config = replace(CONFIG, tie_embeddings=False)
    model, optimizer = fresh(config=config)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model=model, optimizer=optimizer, step=1)
    other, other_optimizer = fresh(seed=5, config=config)
    assert restore(load_checkpoint(path), model=other, optimizer=other_optimizer) == 1
    assert torch.equal(other.lm_head.weight, model.lm_head.weight)
    assert other.lm_head.weight is not other.tok.weight


def test_saving_creates_missing_directories_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    model, optimizer = fresh()
    path = tmp_path / "runs" / "one" / "ckpt.pt"
    save_checkpoint(path, model=model, optimizer=optimizer, step=0)
    assert [entry.name for entry in path.parent.iterdir()] == ["ckpt.pt"]


def test_a_failed_save_keeps_the_previous_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, optimizer = fresh()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model=model, optimizer=optimizer, step=1)
    before = path.read_bytes()

    def explode(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", explode)
    with pytest.raises(OSError, match="disk full"):
        save_checkpoint(path, model=model, optimizer=optimizer, step=2)
    assert path.read_bytes() == before
    assert [entry.name for entry in tmp_path.iterdir()] == ["ckpt.pt"]


def test_negative_steps_are_rejected(tmp_path: Path) -> None:
    model, optimizer = fresh()
    with pytest.raises(CheckpointError, match="non-negative"):
        save_checkpoint(tmp_path / "ckpt.pt", model=model, optimizer=optimizer, step=-1)
    assert list(tmp_path.iterdir()) == []


def test_loading_a_missing_file_fails(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError, match="not a file"):
        load_checkpoint(tmp_path / "nope.pt")


def test_loading_garbage_fails(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.pt"
    path.write_bytes(b"this is not a checkpoint")
    with pytest.raises(CheckpointError, match="cannot read"):
        load_checkpoint(path)


def test_loading_a_foreign_payload_fails(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.pt"
    torch.save({"weights": torch.zeros(1)}, path)
    with pytest.raises(CheckpointError, match="not a checkpoint"):
        load_checkpoint(path)


def test_loading_another_format_version_fails(tmp_path: Path) -> None:
    model, optimizer = fresh()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model=model, optimizer=optimizer, step=0)
    payload = torch.load(path, weights_only=True)
    payload["format"] = FORMAT + 1
    torch.save(payload, path)
    with pytest.raises(CheckpointError, match="format"):
        load_checkpoint(path)


def test_loading_a_corrupt_config_fails(tmp_path: Path) -> None:
    model, optimizer = fresh()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model=model, optimizer=optimizer, step=0)
    payload = torch.load(path, weights_only=True)
    payload["config"]["surprise"] = 1
    torch.save(payload, path)
    with pytest.raises(CheckpointError, match="invalid config"):
        load_checkpoint(path)


def test_loading_an_invalid_step_fails(tmp_path: Path) -> None:
    model, optimizer = fresh()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model=model, optimizer=optimizer, step=0)
    payload = torch.load(path, weights_only=True)
    payload["step"] = -4
    torch.save(payload, path)
    with pytest.raises(CheckpointError, match="invalid step"):
        load_checkpoint(path)


def test_restoring_into_a_different_config_fails(tmp_path: Path) -> None:
    model, optimizer = fresh()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model=model, optimizer=optimizer, step=0)
    other, _ = fresh(config=replace(CONFIG, d_ff=96))
    with pytest.raises(CheckpointError, match="does not match"):
        restore(load_checkpoint(path), model=other)


def test_restoring_a_mismatched_optimizer_fails(tmp_path: Path) -> None:
    model, optimizer = fresh()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model=model, optimizer=optimizer, step=0)
    other = GPT(CONFIG)
    odd = torch.optim.AdamW(other.blocks.parameters(), lr=1e-3)
    with pytest.raises(CheckpointError, match="does not fit"):
        restore(load_checkpoint(path), model=other, optimizer=odd)
