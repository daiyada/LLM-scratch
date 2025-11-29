import pytest
import tiktoken
import torch

from LLM_scratch.config import GPT124M
from LLM_scratch.gpt import GPTModel
from LLM_scratch.utils import (
    generate_txt_simple,
    txt_to_token_ids,
    token_ids_to_txt,
)


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


def test_txt_to_token_ids(tokenizer: tiktoken) -> None:
    """Test txt to token ids."""
    start_context = "Evert effort moves you"
    token_ids = txt_to_token_ids(start_context, tokenizer)
    assert token_ids.shape == torch.Size([1, 5])


def test_token_ids_to_txt(tokenizer: tiktoken) -> None:
    """Test token ids to txt."""
    token_ids = torch.tensor([36, 1851, 3626, 6100, 345])
    generate_txt = token_ids_to_txt(token_ids, tokenizer)
    assert generate_txt == "Evert effort moves you"


def test_generated_txt(tokenizer: tiktoken, gpt124m_config: GPT124M) -> None:
    """Test generated text."""
    torch.manual_seed(123)
    start_context = "Evert effort moves you"
    model = GPTModel(cfg=gpt124m_config)
    token_ids = generate_txt_simple(
        model=model,
        idx=txt_to_token_ids(start_context, tokenizer),
        max_new_tokens=10,
        context_size=gpt124m_config.context_length,
    )
    generated_text = token_ids_to_txt(token_ids, tokenizer)
    assert len(generated_text.split()) == 9
    assert generated_text == (
        "Evert effort moves you fortunate mandatoryicted",
        " VIDEOousse Fan526 WestbrookinchAdmin",
    )
