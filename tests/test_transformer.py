import torch

from LLM_scratch.transformer import TransformerBlock
from LLM_scratch.config import GPT124M


def test_transformer(batch_tensor: torch.Tensor) -> None:
    """Test transformer block."""
    batch, num_tokens, d_in = batch_tensor.shape
    GPT124M.emb_dim = d_in
    GPT124M.n_heads = d_in
    transformer_b = TransformerBlock(cfg=GPT124M)
    output = transformer_b(batch_tensor)

    assert output.shape == torch.Size([batch, num_tokens, GPT124M.emb_dim])
