import logging

import torch

log = logging.getLogger(__name__)

__all__ = ["QuaternionDifference"]


class QuaternionDifference(torch.nn.Module):
    def __init__(self, epsilon: float = 1e-7) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self,
        gt: torch.Tensor,  # (B, T, J, 4)
        pred: torch.Tensor,  # (B,T, J, 4)
    ) -> torch.Tensor:
        angle = torch.acos(
            torch.clamp(
                torch.abs((gt * pred).sum(dim=-1)), -1 + self.epsilon, 1 - self.epsilon
            )
        )
        return (angle**2).sum(dim=-1).sum(dim=-1)
