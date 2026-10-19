from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

import blom.train as train_module
from blom import GPT, ModelConfig, next_token_loss
from blom.data import Shard, eval_batches, train_batch
from blom.optim import build_optimizer
from blom.train import StepStats, TrainError, evaluate, train_step
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
    warmup_steps=4,
    decay_steps=10,
    total_steps=40,
    max_grad_norm=1e9,
)


def pattern_shard(tmp_path: Path, length: int = 2000) -> Shard:
    path = tmp_path / "pattern.bin"
    (np.arange(length) % 8).astype("<u2").tofile(path)
    return Shard(path)


def random_shard(tmp_path: Path, length: int) -> Shard:
    path = tmp_path / "random.bin"
    tokens = np.random.default_rng(0).integers(0, 64, size=length)
    tokens.astype("<u2").tofile(path)
    return Shard(path)


def fresh(seed: int = 0) -> tuple[GPT, torch.optim.AdamW]:
    torch.manual_seed(seed)
    model = GPT(MODEL)
    return model, build_optimizer(model, lr=1e-3)


def capture_grads(
    monkeypatch: pytest.MonkeyPatch, model: GPT, optimizer: torch.optim.AdamW
) -> dict[str, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}

    def spy(*_args: object, **_kwargs: object) -> None:
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None
            captured[name] = parameter.grad.clone()

    monkeypatch.setattr(optimizer, "step", spy)
    return captured


def test_training_learns_a_simple_pattern(tmp_path: Path) -> None:
    shard = pattern_shard(tmp_path)
    model, optimizer = fresh()
    first = train_step(model, optimizer, shard, TRAIN, step=0, device=CPU)
    last = first
    for step in range(1, 40):
        last = train_step(model, optimizer, shard, TRAIN, step=step, device=CPU)
    assert first.loss > 3.5
    assert last.loss < 1.0


def test_stats_describe_the_step(tmp_path: Path) -> None:
    shard = pattern_shard(tmp_path)
    model, optimizer = fresh()
    config = replace(TRAIN, grad_accum=2)
    stats = train_step(model, optimizer, shard, config, step=2, device=CPU)
    assert isinstance(stats, StepStats)
    assert stats.step == 2
    assert stats.tokens == 4 * 2 * 8
    assert stats.lr == config.schedule.lr_at(2)
    assert stats.grad_norm > 0
    assert all(group["lr"] == stats.lr for group in optimizer.param_groups)


def test_the_same_seed_gives_identical_runs(tmp_path: Path) -> None:
    shard = pattern_shard(tmp_path)
    runs = []
    for _ in range(2):
        model, optimizer = fresh()
        runs.append(
            [train_step(model, optimizer, shard, TRAIN, step=s, device=CPU).loss for s in range(3)]
        )
    assert runs[0] == runs[1]


def test_gradient_accumulation_matches_one_big_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shard = pattern_shard(tmp_path)
    config = replace(TRAIN, batch_size=2, grad_accum=2)
    model, optimizer = fresh()
    captured = capture_grads(monkeypatch, model, optimizer)
    stats = train_step(model, optimizer, shard, config, step=3, device=CPU)

    reference, _ = fresh()
    pairs = [
        train_batch(shard, batch_size=2, seq_len=8, seed=config.seed, step=6 + micro)
        for micro in range(2)
    ]
    inputs = torch.cat([pair[0] for pair in pairs])
    targets = torch.cat([pair[1] for pair in pairs])
    loss = next_token_loss(reference(inputs), targets)
    loss.backward()

    assert stats.loss == pytest.approx(loss.item(), rel=1e-5)
    for name, parameter in reference.named_parameters():
        assert parameter.grad is not None
        assert torch.allclose(captured[name], parameter.grad, atol=1e-6), name


def test_gradients_are_clipped_and_the_reported_norm_is_the_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shard = pattern_shard(tmp_path)
    limit = 0.01
    model, optimizer = fresh()
    captured = capture_grads(monkeypatch, model, optimizer)
    stats = train_step(
        model, optimizer, shard, replace(TRAIN, max_grad_norm=limit), step=0, device=CPU
    )
    clipped = torch.linalg.vector_norm(torch.stack([g.norm() for g in captured.values()]))
    assert stats.grad_norm > limit
    assert float(clipped) == pytest.approx(limit, rel=1e-4)


def test_gradients_below_the_limit_are_left_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shard = pattern_shard(tmp_path)
    model, optimizer = fresh()
    captured = capture_grads(monkeypatch, model, optimizer)
    stats = train_step(model, optimizer, shard, TRAIN, step=0, device=CPU)
    norm = torch.linalg.vector_norm(torch.stack([g.norm() for g in captured.values()]))
    assert float(norm) == pytest.approx(stats.grad_norm, rel=1e-5)


def test_a_non_finite_loss_stops_before_the_optimizer_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shard = pattern_shard(tmp_path)
    model, optimizer = fresh()
    model.norm.weight.data.fill_(float("nan"))

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("optimizer stepped")

    monkeypatch.setattr(optimizer, "step", forbidden)
    with pytest.raises(TrainError, match="loss is nan at step 0"):
        train_step(model, optimizer, shard, TRAIN, step=0, device=CPU)


def test_a_non_finite_gradient_norm_stops_before_the_optimizer_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shard = pattern_shard(tmp_path)
    model, optimizer = fresh()

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("optimizer stepped")

    monkeypatch.setattr(optimizer, "step", forbidden)
    monkeypatch.setattr(train_module, "clip_grad_norm_", lambda *_a, **_k: torch.tensor(torch.inf))
    with pytest.raises(TrainError, match="gradient norm is inf"):
        train_step(model, optimizer, shard, TRAIN, step=0, device=CPU)


@pytest.mark.filterwarnings("ignore:Mismatch dtype:UserWarning")
def test_bf16_autocast_runs_and_stays_close_to_full_precision(tmp_path: Path) -> None:
    shard = pattern_shard(tmp_path)
    full, full_optimizer = fresh()
    mixed, mixed_optimizer = fresh()
    exact = train_step(full, full_optimizer, shard, TRAIN, step=0, device=CPU)
    approx = train_step(mixed, mixed_optimizer, shard, TRAIN, step=0, device=CPU, amp=True)
    assert approx.loss == pytest.approx(exact.loss, rel=0.05)


def test_evaluate_averages_over_every_window_weighted_by_tokens(tmp_path: Path) -> None:
    shard = random_shard(tmp_path, length=8 * 13 + 1)
    model, _ = fresh()
    config = replace(TRAIN, batch_size=4, eval_batches=50)
    inputs, targets = next(eval_batches(shard, batch_size=13, seq_len=8))
    with torch.no_grad():
        expected = float(next_token_loss(model(inputs), targets))
    assert evaluate(model, shard, config, device=CPU) == pytest.approx(expected, rel=1e-5)


def test_evaluate_honours_the_batch_limit(tmp_path: Path) -> None:
    shard = random_shard(tmp_path, length=8 * 13 + 1)
    model, _ = fresh()
    config = replace(TRAIN, batch_size=4, eval_batches=1)
    inputs, targets = next(eval_batches(shard, batch_size=4, seq_len=8))
    with torch.no_grad():
        expected = float(next_token_loss(model(inputs), targets))
    limited = evaluate(model, shard, config, device=CPU)
    unlimited = evaluate(model, shard, replace(config, eval_batches=50), device=CPU)
    assert limited == pytest.approx(expected, rel=1e-5)
    assert abs(limited - unlimited) > 1e-4


def test_evaluate_does_not_change_the_training_mode_or_the_weights(tmp_path: Path) -> None:
    shard = pattern_shard(tmp_path)
    model, _ = fresh()
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    model.train()
    evaluate(model, shard, TRAIN, device=CPU)
    assert model.training
    model.eval()
    evaluate(model, shard, TRAIN, device=CPU)
    assert not model.training
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name])
