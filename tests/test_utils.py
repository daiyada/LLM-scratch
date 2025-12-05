from pathlib import Path

import pytest
import tiktoken
import torch
from torch.utils.data import DataLoader

from LLM_scratch.config import GPT124M
from LLM_scratch.gpt import GPTModel
from LLM_scratch.utils import (
    generate_txt_simple,
    txt_to_token_ids,
    token_ids_to_txt,
    prepare_txt_data,
    create_dataloader_v1,
)


def test_prepare_not_existed_txt_data(tmp_path: Path) -> None:
    """Test prepare txt data in case txt file is not existed."""
    expected_start_txt = (
        "I HAD always thought Jack Gisburn rather a cheap genius"
    )
    txt_path = tmp_path / "sample.txt"
    assert not txt_path.exists()
    res_data = prepare_txt_data(str(txt_path))
    assert res_data.startswith(expected_start_txt)
    assert txt_path.exists()


def test_prepare_existed_txt_data(tmp_path: Path) -> None:
    """Test prepare txt data in case txt file is not existed."""
    txt_path = tmp_path / "sample.txt"
    content = "Hello, World!!"
    with open(txt_path, mode="w", encoding="utf-8") as f:
        f.write(content)
    assert txt_path.exists()
    res_data = prepare_txt_data(str(txt_path))
    assert res_data == content


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
        "Evert effort moves you fortunate mandatoryicted"
        " VIDEOousse Fan526 WestbrookinchAdmin"
    )


@pytest.mark.parametrize(
    ("batch_size", "max_length"), [(1, 1), (2, 2), (4, 4)]
)
def test_create_dataloader_v1(
    tmp_path: Path, batch_size: int, max_length
) -> None:
    """Test create dataloader v1 function."""
    txt_path = tmp_path / "sample.txt"
    assert not txt_path.exists()
    txt_data = prepare_txt_data(str(txt_path))
    data_loader = create_dataloader_v1(
        txt_data, batch_size=batch_size, max_length=max_length
    )
    assert type(data_loader) is DataLoader
    data_iter = iter(data_loader)
    first_input_batch, first_target_batch = next(data_iter)
    assert first_input_batch.shape == torch.Size([batch_size, max_length])
    assert first_target_batch.shape == torch.Size([batch_size, max_length])
