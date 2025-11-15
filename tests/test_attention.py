import torch

from attention import CausalAttention


def test_causal_attn(batch_tensor: torch.Tensor) -> None:
    """Test causal attention object."""
    d_out = 2
    batch_size, context_length, d_in = batch_tensor.shape
    ca = CausalAttention(
        d_in=d_in,
        d_out=d_out,
        context_length=context_length,
        dropout=0.0,
        qkv_bias=False,
    )
    context_vecs = ca(batch_tensor)
    assert context_vecs.shape == torch.Size(
        [batch_size, context_length, d_out]
    )
