import torch
import torch.nn as nn


class LayerNorm(nn.Module):

    def __init__(self, emb_dim: int) -> None:
        """Define constructor."""
        super().__init__()
        self._epsilon = 1e-5
        self.__scale = nn.Parameter(torch.ones(emb_dim))
        self.__shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define forward propagation."""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self._epsilon)
        return self.__scale * norm_x + self.__shift
