import torch
from moai.monads.tensor.torch import Flatten


class MDMRepresentation(torch.nn.Module):
    def __init__(self) -> None:
        super(MDMRepresentation, self).__init__()
        self.flatten_pos = Flatten(start_dim=-2)
        self.flatten_rot = Flatten(start_dim=-3)

    def forward(
        self,
        positions: torch.Tensor,
        rotations: torch.Tensor,
        velocities: torch.Tensor,
        linear: torch.Tensor,
        angular: torch.Tensor,
    ) -> torch.Tensor:
        r_a = self.flatten_pos(angular)
        r_x = linear[..., 0]
        r_z = linear[..., 1]
        r_pos_y = positions[:, :, 0, 1].unsqueeze(-1)
        j_p = self.flatten_pos(positions[:, :, 1:, :])
        j_r = self.flatten_rot(rotations[:, :, 1:, :])
        j_v = self.flatten_pos(velocities)

        return torch.cat((r_a, r_x, r_z, r_pos_y, j_p, j_v, j_r), dim=-1)
