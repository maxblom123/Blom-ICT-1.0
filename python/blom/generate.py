import math
from collections.abc import Collection, Iterator, Sequence

import torch
from torch import Tensor

from blom.model import GPT


class GenerationError(ValueError):
    pass


def sample_next(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> int:
    if logits.ndim != 1:
        raise GenerationError(f"logits must be one-dimensional, got shape {tuple(logits.shape)}")
    scores = logits.detach().float().cpu()
    if temperature == 0:
        return int(scores.argmax())
    scores = scores / temperature
    if top_k is not None and top_k < scores.numel():
        threshold = torch.topk(scores, top_k).values[-1]
        scores = scores.masked_fill(scores < threshold, -math.inf)
    if top_p < 1.0:
        ordered, order = torch.sort(scores, descending=True)
        probabilities = torch.softmax(ordered, dim=-1)
        drop = probabilities.cumsum(dim=-1) - probabilities >= top_p
        ordered = ordered.masked_fill(drop, -math.inf)
        scores = torch.full_like(scores, -math.inf).scatter(0, order, ordered)
    probabilities = torch.softmax(scores, dim=-1)
    return int(torch.multinomial(probabilities, 1, generator=generator))


def generate_stream(
    model: GPT,
    prompt: Sequence[int],
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    stop_ids: Collection[int] = (),
    seed: int | None = None,
    amp: bool = False,
) -> Iterator[int]:
    config = model.config
    if not prompt:
        raise GenerationError("the prompt must not be empty")
    if len(prompt) >= config.max_seq:
        raise GenerationError(
            f"the prompt has {len(prompt)} tokens, the model's limit is {config.max_seq - 1}"
        )
    if min(prompt) < 0 or max(prompt) >= config.vocab_size:
        raise GenerationError("the prompt contains token ids outside the vocabulary")
    if max_new_tokens < 1:
        raise GenerationError(f"max_new_tokens must be at least 1, got {max_new_tokens}")
    if not math.isfinite(temperature) or temperature < 0:
        raise GenerationError(f"temperature must be finite and not negative, got {temperature}")
    if top_k is not None and top_k < 1:
        raise GenerationError(f"top_k must be at least 1, got {top_k}")
    if not 0 < top_p <= 1:
        raise GenerationError(f"top_p must be in (0, 1], got {top_p}")
    generator = None if seed is None else torch.Generator().manual_seed(seed)
    budget = min(max_new_tokens, config.max_seq - len(prompt))
    return _stream(
        model,
        list(prompt),
        budget,
        temperature,
        top_k,
        top_p,
        frozenset(stop_ids),
        generator,
        amp,
    )


@torch.no_grad()
def _stream(
    model: GPT,
    prompt: list[int],
    budget: int,
    temperature: float,
    top_k: int | None,
    top_p: float,
    stop_ids: frozenset[int],
    generator: torch.Generator | None,
    amp: bool,
) -> Iterator[int]:
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        caches = model.new_caches()
        tokens = torch.tensor([prompt], dtype=torch.long, device=device)
        for _ in range(budget):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                logits = model(tokens, caches)
            token = sample_next(
                logits[0, -1],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator=generator,
            )
            if token in stop_ids:
                return
            yield token
            tokens = torch.tensor([[token]], dtype=torch.long, device=device)
    finally:
        model.train(was_training)


def generate(
    model: GPT,
    prompt: Sequence[int],
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    stop_ids: Collection[int] = (),
    seed: int | None = None,
    amp: bool = False,
) -> list[int]:
    return list(
        generate_stream(
            model,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_ids=stop_ids,
            seed=seed,
            amp=amp,
        )
    )
