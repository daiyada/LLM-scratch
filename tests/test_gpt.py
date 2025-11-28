import torch

from LLM_scratch.gpt import GPTModel
from LLM_scratch.config import GPT124M


def test_gpt_model(batch_token_idx_tensor: torch.Tensor) -> None:
    """Test gpt model."""
    batch_size, num_tokens = batch_token_idx_tensor.shape
    model = GPTModel(cfg=GPT124M)
    out = model(batch_token_idx_tensor)
    assert out.shape == torch.Size(
        [batch_size, num_tokens, GPT124M.vocab_size]
    )
