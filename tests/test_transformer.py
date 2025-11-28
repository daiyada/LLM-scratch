import torch

from LLM_scratch.transformer import TransformerBlock
from LLM_scratch.config import GPT124M


def test_transformer(
    batch_tensor: torch.Tensor, gpt124m_config: GPT124M
) -> None:
    """Test transformer block."""
    batch, num_tokens, d_in = batch_tensor.shape
    gpt124m_config.emb_dim = d_in
    gpt124m_config.n_heads = d_in
    transformer_b = TransformerBlock(cfg=gpt124m_config)
    output = transformer_b(batch_tensor)

    assert output.shape == torch.Size(
        [batch, num_tokens, gpt124m_config.emb_dim]
    )
