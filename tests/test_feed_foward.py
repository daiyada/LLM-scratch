import torch

from LLM_scratch.feed_forward import GELU, FeedFoward
from LLM_scratch.config import GPT124M


def test_gelu(batch_tensor: torch.Tensor) -> None:
    """Test gelu object"""
    g = GELU()
    gelu = g(batch_tensor)
    assert gelu.shape == batch_tensor.shape


def test_feed_foward(
    batch_tensor: torch.Tensor, gpt124m_config: GPT124M
) -> None:
    """Test feed forward network."""
    batch, num_tokens, _ = batch_tensor.shape
    gpt124m_config.emb_dim = 3
    ffn = FeedFoward(cfg=gpt124m_config)
    out = ffn(batch_tensor)
    assert out.shape == torch.Size([batch, num_tokens, gpt124m_config.emb_dim])
