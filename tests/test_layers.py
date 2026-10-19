import math
from dataclasses import replace

import pytest
import torch
from torch import Tensor

from blom.config import ModelConfig
from blom.layers import Attention, Block, RotaryEmbedding, SwiGLU

CONFIG = ModelConfig(
    vocab_size=100, d_model=64, n_layer=2, n_head=4, n_kv_head=2, d_ff=128, max_seq=32
)


@pytest.fixture(autouse=True)
def seeded() -> None:
    torch.manual_seed(0)


def rope_for(config: ModelConfig) -> RotaryEmbedding:
    return RotaryEmbedding(config.head_dim, config.max_seq, config.rope_theta)


def test_rope_position_zero_is_identity() -> None:
    rope = RotaryEmbedding(16, 8, 10_000.0)
    x = torch.randn(1, 2, 1, 16)
    assert torch.allclose(rope(x), x, atol=1e-6)


def test_rope_preserves_vector_norms() -> None:
    rope = RotaryEmbedding(16, 8, 10_000.0)
    x = torch.randn(2, 3, 8, 16)
    assert torch.allclose(rope(x).norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_rope_offset_matches_full_sequence() -> None:
    rope = RotaryEmbedding(16, 8, 10_000.0)
    x = torch.randn(1, 2, 8, 16)
    full = rope(x)
    tail = rope(x[:, :, 5:], offset=5)
    assert torch.allclose(full[:, :, 5:], tail, atol=1e-6)


def test_rope_scores_depend_only_on_relative_position() -> None:
    rope = RotaryEmbedding(16, 16, 10_000.0)
    q = torch.randn(1, 1, 1, 16)
    k = torch.randn(1, 1, 1, 16)

    def score(q_pos: int, k_pos: int) -> Tensor:
        return (rope(q, q_pos) * rope(k, k_pos)).sum()

    assert torch.allclose(score(3, 1), score(9, 7), atol=1e-5)
    assert not torch.allclose(score(3, 1), score(3, 2), atol=1e-3)


def test_rope_rejects_positions_past_max_seq() -> None:
    rope = RotaryEmbedding(16, 8, 10_000.0)
    with pytest.raises(ValueError, match="exceed max_seq"):
        rope(torch.randn(1, 1, 4, 16), offset=6)


def test_rope_keeps_the_input_dtype() -> None:
    rope = RotaryEmbedding(16, 8, 10_000.0)
    assert rope(torch.randn(1, 1, 4, 16, dtype=torch.bfloat16)).dtype == torch.bfloat16


@pytest.mark.parametrize("qk_norm", [True, False])
@pytest.mark.parametrize("n_kv_head", [1, 2, 4])
def test_attention_matches_a_naive_reference(n_kv_head: int, qk_norm: bool) -> None:
    config = replace(CONFIG, n_kv_head=n_kv_head, qk_norm=qk_norm)
    attention = Attention(config)
    rope = rope_for(config)
    x = torch.randn(2, 10, config.d_model)

    expected = naive_attention(attention, x, rope, config)
    assert torch.allclose(attention(x, rope), expected, atol=1e-5)


def naive_attention(
    attention: Attention, x: Tensor, rope: RotaryEmbedding, config: ModelConfig
) -> Tensor:
    batch, seq, _ = x.shape
    heads, kv_heads, head_dim = config.n_head, config.n_kv_head, config.head_dim
    q = attention.q_proj(x).view(batch, seq, heads, head_dim).transpose(1, 2)
    k = attention.k_proj(x).view(batch, seq, kv_heads, head_dim).transpose(1, 2)
    v = attention.v_proj(x).view(batch, seq, kv_heads, head_dim).transpose(1, 2)
    q, k = rope(attention.q_norm(q)), rope(attention.k_norm(k))
    group = heads // kv_heads
    k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
    scores = q @ k.transpose(-1, -2) / math.sqrt(head_dim)
    mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
    weights = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
    return attention.o_proj((weights @ v).transpose(1, 2).reshape(batch, seq, -1))


def test_attention_is_causal() -> None:
    attention = Attention(CONFIG)
    rope = rope_for(CONFIG)
    x = torch.randn(1, 12, CONFIG.d_model)
    changed = x.clone()
    changed[:, 8:] = torch.randn(1, 4, CONFIG.d_model)
    before = attention(x, rope)
    after = attention(changed, rope)
    assert torch.allclose(before[:, :8], after[:, :8], atol=1e-6)
    assert not torch.allclose(before[:, 8:], after[:, 8:], atol=1e-3)


def test_swiglu_matches_its_formula() -> None:
    mlp = SwiGLU(CONFIG)
    x = torch.randn(2, 5, CONFIG.d_model)
    gate = x @ mlp.gate_proj.weight.T
    up = x @ mlp.up_proj.weight.T
    expected = (torch.nn.functional.silu(gate) * up) @ mlp.down_proj.weight.T
    assert torch.allclose(mlp(x), expected, atol=1e-5)


def test_block_is_the_identity_when_output_projections_are_zero() -> None:
    block = Block(CONFIG)
    torch.nn.init.zeros_(block.attn.o_proj.weight)
    torch.nn.init.zeros_(block.mlp.down_proj.weight)
    x = torch.randn(2, 7, CONFIG.d_model)
    assert torch.equal(block(x, rope_for(CONFIG)), x)


@pytest.mark.parametrize("qk_norm", [True, False])
def test_block_parameter_count_matches_the_config(qk_norm: bool) -> None:
    config = replace(CONFIG, n_layer=1, qk_norm=qk_norm)
    block_params = sum(p.numel() for p in Block(config).parameters())
    outside_blocks = config.vocab_size * config.d_model + config.d_model
    assert block_params == config.param_count - outside_blocks


def test_every_block_parameter_receives_a_gradient() -> None:
    block = Block(CONFIG)
    block(torch.randn(2, 6, CONFIG.d_model), rope_for(CONFIG)).square().sum().backward()
    for name, parameter in block.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
