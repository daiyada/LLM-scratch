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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_out", default=2)
    args = parser.parse_args()

    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],  # Your
            [0.55, 0.87, 0.66],  # journey
            [0.57, 0.85, 0.64],  # starts
            [0.22, 0.58, 0.33],  # with
            [0.77, 0.25, 0.10],  # one
            [0.05, 0.80, 0.55],  # step
        ]
    )
    batch = torch.stack([inputs, inputs], dim=0)
    d_in = inputs.shape[1]
    context_length = batch.shape[1]
    ca = CausalAttention(
        d_in=d_in,
        d_out=args.d_out,
        context_length=context_length,
        dropout=0.0,
        qkv_bias=False,
    )
    context_vecs = ca(batch)
    print(f"context_vecs.shape: {context_vecs.shape}")
