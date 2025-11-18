import argparse

import torch
import torch.nn as nn


class CausalAttention(nn.Module):

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        qkv_bias: bool = False,
    ) -> None:
        """Define constructor."""

        super().__init__()
        self._W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self._W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self._W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self._dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define forward propagation."""
        b, num_tokens, d_in = x.shape
        keys = self._W_key(x)
        queries = self._W_query(x)
        values = self._W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** (1 / 2), dim=-1
        )
        attn_weights = self._dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ):
        """Define constructor."""
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self._d_out = d_out
        self._num_heads = num_heads
        self._head_dim = d_out // num_heads

        self._W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self._W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self._W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Combine output of head by using linear layer
        self._out_proj = nn.Linear(d_out, d_out)
        self._dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(context_length, context_length), diagonal=1
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define forward propagation."""
        b, num_tokens, d_in = x.shape

        queries = self._W_query(x)
        keys = self._W_key(x)
        values = self._W_value(x)

        queries = queries.view(b, num_tokens, self._num_heads, self._head_dim)
        keys = keys.view(b, num_tokens, self._num_heads, self._head_dim)
        values = values.view(b, num_tokens, self._num_heads, self._head_dim)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-2] ** 0.5, dim=-1)
        attn_weights = self._dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self._d_out
        )
        context_vec = self._out_proj(context_vec)

        return context_vec
