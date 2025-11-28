import pytest
import torch


@pytest.fixture(scope="function")
def inputs_tensor() -> torch.Tensor:
    """Example of tensor input."""
    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],  # Your
            [0.55, 0.87, 0.66],  # journey
            [0.57, 0.85, 0.64],  # starts
            [0.22, 0.58, 0.33],  # with
            [0.77, 0.25, 0.10],  # one
            [0.05, 0.80, 0.55],  # step
        ]
    )
    return inputs


@pytest.fixture(scope="function")
def batch_tensor(inputs_tensor: torch.Tensor) -> torch.Tensor:
    """Example of tensor batch consisted of 2 inputs_tensors."""
    batch = torch.stack([inputs_tensor, inputs_tensor], dim=0)
    return batch


@pytest.fixture(scope="function")
def batch_token_idx_tensor() -> torch.Tensor:
    """Example of batch tensor which includes token indices."""
    batch = torch.tensor([[6109, 3626, 6100, 345], [6109, 1110, 6622, 257]])
    return batch
