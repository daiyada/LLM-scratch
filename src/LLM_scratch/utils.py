import torch

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
