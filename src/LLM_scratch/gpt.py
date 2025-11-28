import torch
import torch.nn as nn

from LLM_scratch.config import GPT124M
from LLM_scratch.transformer import TransformerBlock
from LLM_scratch.layer_norm import LayerNorm


class GPTModel(nn.Module):

    def __init__(self, cfg: GPT124M) -> None:
        """Define constructor."""
        super().__init__()
        self._tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self._pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self._drop_emb = nn.Dropout(cfg.drop_rate)

        self._trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )

        self._final_norm = LayerNorm(cfg.emb_dim)
        self._out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """Define forward propagation."""
        _, seq_len = in_idx.shape
        tok_embeds = self._tok_emb(in_idx)
        pos_embeds = self._pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self._drop_emb(x)
        x = self._trf_blocks(x)
        x = self._final_norm(x)
        logits = self._out_head(x)
        return logits
