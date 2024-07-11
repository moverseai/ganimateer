import torch

__all__ = [
    "SimpleVelocity",
]


class SimpleVelocity(torch.nn.Module):  # NOTE: check finite difference
    def __init__(self, zero_pad: bool = False, return_norm: bool = True) -> None:
        super().__init__()
        self.zero_pad = zero_pad
        self.return_norm = return_norm

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        dp = (
            positions[1:, ...] - positions[:-1, ...]
            if positions.ndim <= 3
            else positions[:, 1:, ...] - positions[:, :-1, ...]
        )
        if self.return_norm:
            norm = torch.norm(dp, dim=-1)
            if self.zero_pad:
                pad = (
                    torch.zeros_like(norm[:1, :])
                    if positions.ndim <= 3
                    else torch.zeros_like(norm[:, :1, :])
                )
                norm = torch.cat([pad, norm], dim=0 if positions.ndim <= 3 else 1)
            return norm
        else:
            if self.zero_pad:
                pad = (
                    torch.zeros_like(dp[:1, :])
                    if positions.ndim <= 3
                    else torch.zeros_like(dp[:, :1, :])
                )
                res = torch.cat([pad, dp], dim=0 if positions.ndim <= 3 else 1)
            else:
                res = dp
            return res
