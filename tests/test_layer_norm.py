import torch

from LLM_scratch.layer_norm import LayerNorm


def test_layer_norm(batch_tensor: torch.Tensor) -> None:
    """Test layer norm object."""
    batch, num_tokens, emb_dim = batch_tensor.shape
    ln = LayerNorm(emb_dim=emb_dim)
    out_ln = ln(batch_tensor)
    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
    assert torch.allclose(
        mean.detach(),
        torch.zeros(batch, num_tokens, 1),
        atol=1e-4,
    )
    assert torch.allclose(
        var.detach(),
        torch.ones(batch, num_tokens, 1),
        atol=1e-3,
    )
