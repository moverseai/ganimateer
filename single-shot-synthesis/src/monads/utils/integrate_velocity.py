import torch

__all__ = ["IntegrateVelocity"]


class IntegrateVelocity(torch.nn.Module):
    def __init__(self, ignore_y: bool = True) -> None:
        super().__init__()
        self.ignore_y = ignore_y

    def forward(
        self,
        velocity: torch.Tensor,  # [B, T, 3] (predicted)
        position: torch.Tensor,  # [B, T, 3] (gt)
    ) -> torch.Tensor:
        mask = [0, 1, 2] if not self.ignore_y else [0, 2]
        pos = velocity.clone()
        pos[..., 0, :] = position[..., 0, :]
        pos[..., mask] = torch.cumsum(pos[..., mask], dim=-2)
        return pos
