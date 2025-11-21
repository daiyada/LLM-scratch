import torch
import torch.nn as nn

from LLM_scratch.attention import MultiHeadAttention
from LLM_scratch.config import GPT124M
from LLM_scratch.feed_forward import FeedFoward
from LLM_scratch.layer_norm import LayerNorm


class TransformerBlock(nn.Module):

    def __init__(self, cfg: GPT124M):
        """Define constructor."""
        super().__init__()
        self._attention = MultiHeadAttention(
            d_in=cfg.emb_dim,
            d_out=cfg.emb_dim,
            context_length=cfg.context_length,
            num_heads=cfg.n_heads,
            dropout=cfg.drop_rate,
            qkv_bias=cfg.qkv_bias,
        )
        self._ff = FeedFoward(cfg=cfg)
        self._norm1 = LayerNorm(emb_dim=cfg.emb_dim)
        self._norm2 = LayerNorm(emb_dim=cfg.emb_dim)
        self._drop_shortcut = nn.Dropout(cfg.drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define forward propagation."""
        shortcut = x
        x = self._norm1(x)
        x = self._attention(x)
        x = self._drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self._norm2(x)
        x = self._ff(x)
        x = self._drop_shortcut(x)
        x = x + shortcut

        return x
