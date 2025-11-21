import torch

from LLM_scratch.gelu import GELU


def test_gelu(batch_tensor: torch.Tensor) -> None:
    """Test gelu object"""
    g = GELU()
    gelu = g(batch_tensor)
    assert torch.allclose(
        gelu,
        torch.tensor(
            [
                [
                    [0.2914, 0.0840, 0.7765],
                    [0.4017, 0.7526, 0.5140],
                    [0.4213, 0.7289, 0.4928],
                    [0.1295, 0.4313, 0.2094],
                    [0.6357, 0.1503, 0.0540],
                    [0.0260, 0.6702, 0.4017],
                ],
                [
                    [0.2914, 0.0840, 0.7765],
                    [0.4017, 0.7526, 0.5140],
                    [0.4213, 0.7289, 0.4928],
                    [0.1295, 0.4313, 0.2094],
                    [0.6357, 0.1503, 0.0540],
                    [0.0260, 0.6702, 0.4017],
                ],
            ]
        ),
        atol=1e-4,
    )
