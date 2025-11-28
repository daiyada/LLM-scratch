import pytest
import tiktoken
import torch

from LLM_scratch.config import GPT124M
from LLM_scratch.gpt import GPTModel
from LLM_scratch.utils import generate_txt_simple


@pytest.mark.parametrize("max_new_tokens", [0, 1, 5])
def test_generate_txt_simple(
    tokenizer: tiktoken, gpt124m_config: GPT124M, max_new_tokens: int
) -> None:
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    # add batch: (batch, token_num)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    model = GPTModel(cfg=gpt124m_config)
    model.eval()
    out = generate_txt_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=max_new_tokens,
        context_size=gpt124m_config.context_length,
    )
    assert len(out[0]) == len(encoded) + max_new_tokens
