import torch
import torch.nn.functional as F
from torch import Tensor, nn

from blom.config import ModelConfig


class RotaryEmbedding(nn.Module):
    cos: Tensor
    sin: Tensor

    def __init__(self, head_dim: int, max_seq: int, theta: float) -> None:
        super().__init__()
        exponents = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        angles = torch.outer(torch.arange(max_seq, dtype=torch.float32), 1.0 / theta**exponents)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        seq = x.shape[-2]
        if offset < 0 or offset + seq > self.cos.shape[0]:
            raise ValueError(
                f"positions {offset}..{offset + seq} exceed max_seq {self.cos.shape[0]}"
            )
        cos = self.cos[offset : offset + seq]
        sin = self.sin[offset : offset + seq]
        half = x.shape[-1] // 2
        first, second = x[..., :half].float(), x[..., half:].float()
        rotated = torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)
        return rotated.to(x.dtype)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.head_dim
        d_model = config.d_model
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, config.kv_dim, bias=False)
        self.v_proj = nn.Linear(d_model, config.kv_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm: nn.Module = nn.Identity()
        self.k_norm: nn.Module = nn.Identity()
        if config.qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=config.norm_eps)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=config.norm_eps)

    def forward(self, x: Tensor, rope: RotaryEmbedding, offset: int = 0) -> Tensor:
        batch, seq, _ = x.shape
        q = self.q_proj(x).view(batch, seq, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.n_kv_head, self.head_dim).transpose(1, 2)
        q = rope(self.q_norm(q), offset)
        k = rope(self.k_norm(k), offset)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=self.n_kv_head != self.n_head
        )
        return self.o_proj(y.transpose(1, 2).reshape(batch, seq, -1))


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = Attention(config)
        self.mlp_norm = nn.RMSNorm(config.d_model, eps=config.norm_eps)
        self.mlp = SwiGLU(config)

    def forward(self, x: Tensor, rope: RotaryEmbedding, offset: int = 0) -> Tensor:
        x = x + self.attn(self.attn_norm(x), rope, offset)
        return x + self.mlp(self.mlp_norm(x))
