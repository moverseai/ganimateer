import torch

__all__ = ["NoiseScale"]


class NoiseScale(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, target: torch.Tensor, reconstructed: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.MSELoss()(target, reconstructed) ** 0.5
