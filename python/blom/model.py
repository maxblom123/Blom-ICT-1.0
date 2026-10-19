import math

import torch.nn.functional as F
from torch import Tensor, nn

from blom.config import ModelConfig
from blom.layers import Block, RotaryEmbedding

INIT_STD = 0.02
IGNORE_INDEX = -100


class GPT(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.tok = nn.Embedding(config.vocab_size, config.d_model)
        self.rope = RotaryEmbedding(config.head_dim, config.max_seq, config.rope_theta)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.norm = nn.RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self._init_weights()
        if config.tie_embeddings:
            self.lm_head.weight = self.tok.weight

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear | nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=INIT_STD)
        residual_std = INIT_STD / math.sqrt(2 * self.config.n_layer)
        for module in self.modules():
            if isinstance(module, Block):
                nn.init.normal_(module.attn.o_proj.weight, mean=0.0, std=residual_std)
                nn.init.normal_(module.mlp.down_proj.weight, mean=0.0, std=residual_std)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 2:
            raise ValueError(f"tokens must be (batch, seq), got shape {tuple(tokens.shape)}")
        if tokens.shape[1] > self.config.max_seq:
            raise ValueError(f"sequence length {tokens.shape[1]} exceeds {self.config.max_seq}")
        x = self.tok(tokens)
        for block in self.blocks:
            x = block(x, self.rope)
        return self.lm_head(self.norm(x))


def next_token_loss(logits: Tensor, targets: Tensor) -> Tensor:
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )
