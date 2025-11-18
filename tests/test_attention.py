import pytest
import torch

from attention import CausalAttention, MultiHeadAttention


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

def test_positive_multi_head_attn(batch_tensor: torch.Tensor) -> None:
    """Test multi head attention object in case of positive."""
    d_out = 2
    batch_size, context_length, d_in = batch_tensor.shape
    mha = MultiHeadAttention(
        d_in=d_in,
        d_out=d_out,
        context_length=context_length,
        dropout=0.,
        num_heads=2,
        qkv_bias=False
    )
    context_vecs = mha(batch_tensor)
    assert context_vecs.shape == torch.Size(
        [batch_size, context_length, d_out]
    )

@pytest.mark.parametrize(
        "d_out, num_heads",
        [
            (3, 2),
            (2, 3),
        ]
)
def test_negative_multi_head_attn(
    batch_tensor: torch.Tensor, d_out: int, num_heads: int
) -> None:
    """Test multi head attention object in case of negative."""
    _, context_length, d_in = batch_tensor.shape
    with pytest.raises(AssertionError) as e:
        mha = MultiHeadAttention(
            d_in=d_in,
            d_out=d_out,
            context_length=context_length,
            dropout=0.,
            num_heads=num_heads,
            qkv_bias=False
        )
    assert str(e.value) == "d_out must be divisible by num_heads"