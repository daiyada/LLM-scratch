import os
import requests
import torch
import tiktoken
from torch.utils.data import DataLoader

from LLM_scratch.gpt import GPTModel
from LLM_scratch.data import GPTDatasetV1


def prepare_txt_data(file_path: str) -> str:
    """
    Prepare txt data. If file path you designate as argument is not existed,
    download txt file from url (https://raw.githubusercontent.com/rasbt/
    LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt) and
    read it.
    """
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/"
        "main/ch02/01_main-chapter-code/the-verdict.txt"
    )
    if not os.path.exists(file_path):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        txt_data = response.text
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(txt_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            txt_data = file.read()
    return txt_data


def generate_txt_simple(
    model: GPTModel, idx: int, max_new_tokens: int, context_size: int
) -> torch.Tensor:
    """Generate idx related to new txt."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        # (batch, n_token, vocab_size) -> (batch, vocab_size)
        logits = logits[:, -1, :]
        # (batch, vocab_size)
        probas = torch.softmax(logits, dim=-1)
        # (batch, 1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def txt_to_token_ids(text: str, tokenizer: tiktoken) -> torch.Tensor:
    """Transform text to token ids."""
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    # add batch dimension
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_txt(token_ids: torch.Tensor, tokenizer: tiktoken) -> str:
    """Transform token ids to text."""
    # remove batch dimension
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def create_dataloader_v1(
    txt: str,
    batch_size: int = 4,
    max_length: int = 26,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create data loader by using GPTDatasetV1 object."""
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return data_loader
