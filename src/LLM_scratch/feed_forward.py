import torch
import torch.nn as nn

from LLM_scratch.config import GPT124M


class GELU(nn.Module):

    def __init__(self) -> None:
        """Define constructor."""
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define forward propagation."""
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.44715 * torch.pow(x, 3))
                )
            )
        )


class FeedFoward(nn.Module):

    def __init__(self, cfg: GPT124M) -> None:
        """Define constructor."""
        super().__init__()
        self._layers = nn.Sequential(
            nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim),
            GELU(),
            nn.Linear(4 * cfg.emb_dim, cfg.emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define forward propagation."""
        return self._layers(x)
