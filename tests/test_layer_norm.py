import torch

from LLM_scratch.layer_norm import LayerNorm


def test_layer_norm(batch_example: torch.Tensor) -> None:
    """Test layer norm object."""
    emb_dim = 5
    ln = LayerNorm(emb_dim=emb_dim)
    out_ln = ln(batch_example)
    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
    assert torch.allclose(
        mean.detach(), torch.tensor([[-2.9802e-08], [0.0000e00]]), atol=1e-4
    )
    assert torch.allclose(
        var.detach(), torch.tensor([[1.0000], [1.0000]]), atol=1e-4
    )
