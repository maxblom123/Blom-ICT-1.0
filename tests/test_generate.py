import math
from collections import Counter

import pytest
import torch

from blom import GPT, ModelConfig
from blom.generate import GenerationError, generate, generate_stream, sample_next

CONFIG = ModelConfig(
    vocab_size=64, d_model=32, n_layer=2, n_head=4, n_kv_head=2, d_ff=64, max_seq=16
)


@pytest.fixture(autouse=True)
def seeded() -> None:
    torch.manual_seed(0)


def model_with_opinions() -> GPT:
    model = GPT(CONFIG)
    for parameter in model.parameters():
        if parameter.ndim >= 2:
            torch.nn.init.normal_(parameter, std=0.2)
    return model


def greedy_reference(model: GPT, prompt: list[int], count: int) -> list[int]:
    ids = list(prompt)
    with torch.no_grad():
        for _ in range(count):
            logits = model(torch.tensor([ids]))
            ids.append(int(logits[0, -1].argmax()))
    return ids[len(prompt) :]


def test_greedy_generation_matches_recomputing_everything_each_step() -> None:
    model = model_with_opinions()
    prompt = [3, 14, 15, 9]
    assert generate(model, prompt, max_new_tokens=10, temperature=0) == greedy_reference(
        model, prompt, 10
    )


def test_generation_is_reproducible_with_a_seed_and_varies_without_one() -> None:
    model = model_with_opinions()
    first = generate(model, [1, 2], max_new_tokens=12, seed=7)
    second = generate(model, [1, 2], max_new_tokens=12, seed=7)
    other = generate(model, [1, 2], max_new_tokens=12, seed=8)
    assert first == second
    assert first != other


def test_the_stop_token_ends_generation_and_is_not_returned() -> None:
    model = model_with_opinions()
    free = generate(model, [1, 2], max_new_tokens=8, temperature=0)
    stop = free[3]
    cut = free.index(stop)
    assert generate(model, [1, 2], max_new_tokens=8, temperature=0, stop_ids={stop}) == free[:cut]


def test_generation_never_exceeds_the_context_window() -> None:
    model = model_with_opinions()
    prompt = list(range(10))
    out = generate(model, prompt, max_new_tokens=100, temperature=0)
    assert len(out) == CONFIG.max_seq - len(prompt)


def test_max_new_tokens_is_respected() -> None:
    model = model_with_opinions()
    assert len(generate(model, [1], max_new_tokens=5, temperature=0)) == 5


def test_the_stream_yields_tokens_one_at_a_time_and_leaves_grad_mode_alone() -> None:
    model = model_with_opinions()
    produced = []
    for token in generate_stream(model, [1, 2, 3], max_new_tokens=4, temperature=0):
        assert torch.is_grad_enabled()
        produced.append(token)
    assert produced == generate(model, [1, 2, 3], max_new_tokens=4, temperature=0)


def test_the_model_mode_is_restored_after_generation() -> None:
    model = model_with_opinions()
    model.train()
    generate(model, [1], max_new_tokens=2)
    assert model.training
    model.eval()
    generate(model, [1], max_new_tokens=2)
    assert not model.training


def test_the_prompt_is_validated_before_anything_runs() -> None:
    model = model_with_opinions()
    with pytest.raises(GenerationError, match="empty"):
        generate_stream(model, [], max_new_tokens=1)
    with pytest.raises(GenerationError, match="limit is 15"):
        generate_stream(model, list(range(16)), max_new_tokens=1)
    with pytest.raises(GenerationError, match="outside the vocabulary"):
        generate_stream(model, [64], max_new_tokens=1)
    with pytest.raises(GenerationError, match="outside the vocabulary"):
        generate_stream(model, [-1], max_new_tokens=1)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_new_tokens": 0}, "max_new_tokens"),
        ({"temperature": -1.0}, "temperature"),
        ({"temperature": math.nan}, "temperature"),
        ({"top_k": 0}, "top_k"),
        ({"top_p": 0.0}, "top_p"),
        ({"top_p": 1.5}, "top_p"),
    ],
)
def test_sampling_settings_are_validated(kwargs: dict[str, float], message: str) -> None:
    model = model_with_opinions()
    settings = {"max_new_tokens": 3, **kwargs}
    with pytest.raises(GenerationError, match=message):
        generate_stream(model, [1], **settings)


LOGITS = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0, -1.0])


def draw(
    count: int, *, temperature: float = 1.0, top_k: int | None = None, top_p: float = 1.0
) -> Counter[int]:
    generator = torch.Generator().manual_seed(0)
    return Counter(
        sample_next(LOGITS, temperature=temperature, top_k=top_k, top_p=top_p, generator=generator)
        for _ in range(count)
    )


def test_zero_temperature_is_greedy() -> None:
    assert sample_next(LOGITS, temperature=0) == 0


def test_top_k_never_samples_outside_the_k_best() -> None:
    assert set(draw(500, top_k=2)) == {0, 1}
    assert set(draw(500, top_k=1)) == {0}


def test_top_k_larger_than_the_vocabulary_is_harmless() -> None:
    assert set(draw(2000, top_k=100)) == set(range(6))


def test_top_p_keeps_the_smallest_set_reaching_the_mass() -> None:
    probabilities = torch.softmax(LOGITS, dim=-1)
    nucleus = int((probabilities.cumsum(0) - probabilities < 0.8).sum())
    seen = set(draw(1000, top_p=0.8))
    assert seen <= set(range(nucleus))
    assert len(seen) == nucleus
    assert set(draw(200, top_p=1e-6)) == {0}


def test_top_p_stops_exactly_when_the_mass_is_reached() -> None:
    generator = torch.Generator().manual_seed(0)
    flat = torch.zeros(2)
    drawn = {sample_next(flat, top_p=0.5, generator=generator) for _ in range(50)}
    assert len(drawn) == 1


def test_temperature_reshapes_the_distribution() -> None:
    cold = draw(1000, temperature=0.3)
    hot = draw(1000, temperature=3.0)
    assert cold[0] > hot[0]
    assert len(hot) >= len(cold)


def test_sampling_matches_the_softmax_probabilities() -> None:
    counts = draw(20000)
    expected = torch.softmax(LOGITS, dim=-1)
    for token in range(6):
        assert counts[token] / 20000 == pytest.approx(float(expected[token]), abs=0.015)


def test_sampling_needs_one_dimensional_logits() -> None:
    with pytest.raises(GenerationError, match="one-dimensional"):
        sample_next(LOGITS.view(1, -1))
