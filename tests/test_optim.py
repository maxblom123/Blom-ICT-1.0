from dataclasses import replace

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from blom import GPT, ModelConfig
from blom.layers import Block
from blom.optim import Schedule, ScheduleError, build_optimizer, set_lr

SCHEDULE = Schedule(peak_lr=1e-3, min_lr=1e-4, warmup_steps=10, decay_steps=20, total_steps=100)

CONFIG = ModelConfig(
    vocab_size=64, d_model=32, n_layer=2, n_head=4, n_kv_head=2, d_ff=64, max_seq=16
)


def test_warmup_ramps_linearly_to_the_peak() -> None:
    assert SCHEDULE.lr_at(0) == pytest.approx(1e-4)
    assert SCHEDULE.lr_at(4) == pytest.approx(5e-4)
    assert SCHEDULE.lr_at(9) == pytest.approx(1e-3)


def test_the_stable_phase_holds_the_peak() -> None:
    assert {SCHEDULE.lr_at(step) for step in range(10, 80)} == {1e-3}


def test_decay_is_linear_and_ends_exactly_at_the_floor() -> None:
    assert SCHEDULE.lr_at(80) == pytest.approx(1e-3 - 0.9e-3 / 20)
    assert SCHEDULE.lr_at(99) == pytest.approx(1e-4)


def test_decay_to_zero_is_supported() -> None:
    schedule = replace(SCHEDULE, min_lr=0.0)
    assert schedule.lr_at(99) == 0.0


@pytest.mark.parametrize("step", [-1, 100, 1000])
def test_steps_outside_the_run_are_rejected(step: int) -> None:
    with pytest.raises(ScheduleError, match="outside"):
        SCHEDULE.lr_at(step)


@pytest.mark.parametrize(
    "overrides",
    [
        {"warmup_steps": 0},
        {"decay_steps": 0},
        {"warmup_steps": 90, "decay_steps": 20},
        {"min_lr": 2e-3},
        {"min_lr": -1.0},
        {"peak_lr": 0.0, "min_lr": 0.0},
    ],
)
def test_invalid_schedules_are_rejected(overrides: dict[str, float]) -> None:
    with pytest.raises(ScheduleError):
        replace(SCHEDULE, **overrides)


@given(
    warmup=st.integers(1, 50),
    decay=st.integers(1, 50),
    stable=st.integers(0, 50),
    peak=st.floats(1e-6, 1.0),
    floor=st.floats(0.0, 1.0),
)
def test_schedule_shape_holds_for_any_valid_schedule(
    warmup: int, decay: int, stable: int, peak: float, floor: float
) -> None:
    schedule = Schedule(
        peak_lr=peak,
        min_lr=floor * peak,
        warmup_steps=warmup,
        decay_steps=decay,
        total_steps=warmup + stable + decay,
    )
    rates = [schedule.lr_at(step) for step in range(schedule.total_steps)]
    assert all(0 <= rate <= peak * (1 + 1e-12) for rate in rates)
    assert all(rate >= schedule.min_lr for rate in rates[warmup:])
    rising = rates[: warmup + stable]
    assert rising == sorted(rising)
    falling = rates[warmup + stable - 1 :]
    assert falling == sorted(falling, reverse=True)
    assert rates[-1] == pytest.approx(schedule.min_lr)
    assert max(rates) == pytest.approx(peak)


def test_every_parameter_is_in_exactly_one_group() -> None:
    for tie in (True, False):
        model = GPT(replace(CONFIG, tie_embeddings=tie))
        optimizer = build_optimizer(model, lr=1e-3)
        grouped = [id(p) for group in optimizer.param_groups for p in group["params"]]
        assert len(grouped) == len(set(grouped))
        assert set(grouped) == {id(p) for p in model.parameters()}


def test_only_weight_matrices_get_weight_decay() -> None:
    model = GPT(replace(CONFIG, tie_embeddings=False))
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.1)
    decayed = {id(p) for g in optimizer.param_groups if g["weight_decay"] > 0 for p in g["params"]}
    for name, parameter in model.named_parameters():
        expected = parameter.ndim >= 2 and name != "tok.weight"
        assert (id(parameter) in decayed) == expected, name


def test_decay_actually_shrinks_decayed_weights_only() -> None:
    model = GPT(CONFIG)
    optimizer = build_optimizer(model, lr=0.1, weight_decay=0.5)
    block = model.blocks[0]
    assert isinstance(block, Block)
    before_matrix = block.mlp.up_proj.weight.detach().clone()
    before_norm = model.norm.weight.detach().clone()
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    assert torch.allclose(block.mlp.up_proj.weight, before_matrix * (1 - 0.1 * 0.5))
    assert torch.equal(model.norm.weight, before_norm)


def test_set_lr_updates_every_group() -> None:
    optimizer = build_optimizer(GPT(CONFIG), lr=1e-3)
    set_lr(optimizer, 5e-4)
    assert [group["lr"] for group in optimizer.param_groups] == [5e-4, 5e-4]
