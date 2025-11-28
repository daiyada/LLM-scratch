import torch

from LLM_scratch.gpt import GPTModel
from LLM_scratch.config import GPT124M


def test_gpt_model(
    batch_token_idx_tensor: torch.Tensor, gpt124m_config: GPT124M
) -> None:
    """Test gpt model."""
    batch_size, num_tokens = batch_token_idx_tensor.shape
    model = GPTModel(cfg=gpt124m_config)
    out = model(batch_token_idx_tensor)
    assert out.shape == torch.Size(
        [batch_size, num_tokens, gpt124m_config.vocab_size]
    )
