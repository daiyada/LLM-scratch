from pathlib import Path

import pytest
import tiktoken
import torch

from LLM_scratch.utils import prepare_txt_data
from LLM_scratch.data import GPTDatasetV1


@pytest.mark.parametrize(
    "max_length, stride",
    [(10, 1), (5, 2), (0, 1)],
)
def test_gpt_dataset_v1(
    tmp_path: Path, tokenizer: tiktoken, max_length: int, stride: int
) -> None:
    """Test prepare gpt dataset of version 1."""
    txt_path = tmp_path / "sample.txt"
    assert not txt_path.exists()
    txt_data = prepare_txt_data(str(txt_path))
    dataset_v1 = GPTDatasetV1(
        txt=txt_data, tokenizer=tokenizer, max_length=max_length, stride=stride
    )
    token_ids = tokenizer.encode(txt_data)
    assert (len(token_ids) - max_length) / stride == len(dataset_v1)
    input, target = dataset_v1[0]
    assert input.shape == torch.Size([max_length])
    assert target.shape == torch.Size([max_length])
