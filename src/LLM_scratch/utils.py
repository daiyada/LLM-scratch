import torch
import tiktoken

from LLM_scratch.gpt import GPTModel


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
