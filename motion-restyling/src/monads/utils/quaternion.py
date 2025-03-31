import logging

import torch

log = logging.getLogger(__name__)


class QuaternionNorm(torch.nn.Module):
    def __init__(self, dim: int = -1, keepdim: bool = True):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, quaternion: torch.Tensor) -> torch.Tensor:
        return quaternion / (
            torch.norm(quaternion, dim=self.dim, keepdim=self.keepdim) + 1e-7
        )
