import math
from dataclasses import replace

import pytest
import torch

from blom import GPT, IGNORE_INDEX, ModelConfig, next_token_loss
from blom.layers import Block

CONFIG = ModelConfig(
    vocab_size=64, d_model=32, n_layer=2, n_head=4, n_kv_head=2, d_ff=64, max_seq=16
)


@pytest.fixture(autouse=True)
def seeded() -> None:
    torch.manual_seed(0)


def random_tokens(batch: int, length: int, vocab: int = 64) -> torch.Tensor:
    return torch.randint(0, vocab, (batch, length))


@pytest.mark.parametrize("tie", [True, False])
@pytest.mark.parametrize("qk_norm", [True, False])
def test_parameter_count_matches_the_config(tie: bool, qk_norm: bool) -> None:
    config = replace(CONFIG, tie_embeddings=tie, qk_norm=qk_norm)
    model = GPT(config)
    assert sum(p.numel() for p in model.parameters()) == config.param_count


def test_tied_embeddings_share_one_tensor() -> None:
    tied = GPT(CONFIG)
    untied = GPT(replace(CONFIG, tie_embeddings=False))
    assert tied.lm_head.weight is tied.tok.weight
    assert untied.lm_head.weight is not untied.tok.weight


def test_output_has_one_logit_per_vocabulary_entry() -> None:
    logits = GPT(CONFIG)(random_tokens(3, 10))
    assert logits.shape == (3, 10, CONFIG.vocab_size)
    assert logits.dtype == torch.float32


def test_initial_loss_is_close_to_uniform_guessing() -> None:
    config = replace(CONFIG, vocab_size=1000, d_model=64, d_ff=128)
    model = GPT(config)
    tokens = random_tokens(8, 16, vocab=1000)
    loss = next_token_loss(model(tokens[:, :-1]), tokens[:, 1:])
    assert abs(loss.item() - math.log(1000)) < 0.1


def test_residual_projections_start_smaller_than_other_weights() -> None:
    config = replace(CONFIG, n_layer=8, d_model=128, n_head=8, n_kv_head=4, d_ff=256)
    model = GPT(config)
    block = model.blocks[0]
    assert isinstance(block, Block)
    expected = 0.02 / math.sqrt(2 * config.n_layer)
    assert block.attn.o_proj.weight.std().item() == pytest.approx(expected, rel=0.1)
    assert block.mlp.down_proj.weight.std().item() == pytest.approx(expected, rel=0.1)
    assert block.attn.q_proj.weight.std().item() == pytest.approx(0.02, rel=0.1)


def test_the_same_seed_gives_the_same_model() -> None:
    torch.manual_seed(123)
    first = GPT(CONFIG)
    torch.manual_seed(123)
    second = GPT(CONFIG)
    for (name, a), (_, b) in zip(first.named_parameters(), second.named_parameters(), strict=True):
        assert torch.equal(a, b), name


def test_the_whole_model_is_causal() -> None:
    model = GPT(CONFIG).eval()
    tokens = random_tokens(2, 12)
    changed = tokens.clone()
    changed[:, 7:] = random_tokens(2, 5)
    with torch.no_grad():
        before = model(tokens)
        after = model(changed)
    assert torch.allclose(before[:, :7], after[:, :7], atol=1e-6)
    assert not torch.allclose(before[:, 7:], after[:, 7:], atol=1e-3)


def test_rejects_bad_input_shapes() -> None:
    model = GPT(CONFIG)
    with pytest.raises(ValueError, match="batch, seq"):
        model(random_tokens(1, 4)[0])
    with pytest.raises(ValueError, match="exceeds"):
        model(random_tokens(1, CONFIG.max_seq + 1))


def test_ignored_targets_do_not_contribute_to_the_loss() -> None:
    logits = GPT(CONFIG)(random_tokens(2, 8))
    targets = random_tokens(2, 8)
    masked = targets.clone()
    masked[:, :4] = IGNORE_INDEX
    expected = next_token_loss(logits[:, 4:], targets[:, 4:])
    assert torch.allclose(next_token_loss(logits, masked), expected, atol=1e-6)


def test_every_parameter_receives_a_gradient() -> None:
    model = GPT(CONFIG)
    tokens = random_tokens(4, 12)
    next_token_loss(model(tokens[:, :-1]), tokens[:, 1:]).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_the_model_can_memorise_one_batch() -> None:
    model = GPT(CONFIG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    tokens = random_tokens(4, 16)
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    first = next_token_loss(model(inputs), targets).item()
    for _ in range(80):
        optimizer.zero_grad()
        next_token_loss(model(inputs), targets).backward()
        optimizer.step()
    last = next_token_loss(model(inputs), targets).item()
    assert first == pytest.approx(math.log(CONFIG.vocab_size), abs=0.2)
    assert last < 0.25 * first
