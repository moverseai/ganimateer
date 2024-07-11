import torch

__all__ = ["RootFeatures"]


class RootFeatures(torch.nn.Module):
    def __init__(
        self,
        use_velocity: bool = True,
        keep_y_position: bool = True,
    ) -> None:
        super().__init__()
        self.use_velocity = use_velocity
        self.keep_y_position = keep_y_position

    def forward(
        self,
        position: torch.Tensor,  # [B, T, 3]
    ) -> torch.Tensor:
        if not self.use_velocity:
            return position
        else:
            mask = [-3, -2, -1] if not self.keep_y_position else [-3, -1]
            pos = position.clone()
            pos[:, 1:, mask] = pos[:, 1:, mask] - pos[:, :-1, mask]
            pos[:, 0, mask] = pos[:, 1, mask]
            return pos
