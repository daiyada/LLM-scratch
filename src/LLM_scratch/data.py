from typing import Tuple

import tiktoken
import torch
from torch.utils.data import Dataset


class GPTDatasetV1(Dataset):

    def __init__(
        self, txt: str, tokenizer: tiktoken, max_length: int, stride: int
    ):
        """Define constructor."""
        super().__init__()
        self._input_ids = []
        self._target_ids = []
        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self._input_ids.append(torch.tensor(input_chunk))
            self._target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        """Get length of self._input_ids."""
        return len(self._input_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get tuple consisted of target input id and target id."""
        return self._input_ids[idx], self._target_ids[idx]
