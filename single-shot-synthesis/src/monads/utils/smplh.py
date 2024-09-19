import typing

import numpy as np
import torch

__all__ = ["ToSMPLH"]


class ToSMPLH(torch.nn.Module):
    def __init__(
        self,
        mapping: typing.Sequence[typing.Sequence[int]],
        from_joint_number: int,
    ) -> None:
        super().__init__()
        rotation_indices = []
        position_regressors = []
        for indices in mapping:
            regressor = torch.zeros(from_joint_number, dtype=torch.float32)
            if len(indices) > 1:
                rotation_indices.append(from_joint_number)
            else:
                rotation_indices.append(indices[0])
            for index in indices:
                regressor[index] = 1.0 / len(indices)
            position_regressors.append(regressor)
        self.register_buffer("A", torch.stack(position_regressors))
        self.register_buffer("I", torch.LongTensor(rotation_indices))

    def forward(
        self,
        positions: torch.Tensor,
        rotations: torch.Tensor,
    ) -> typing.Mapping[str, torch.Tensor]:
        pos = torch.einsum("btjc,lj->btlc", positions, self.A)
        id = torch.eye(3).to(rotations)[:, :2][np.newaxis, np.newaxis, np.newaxis]
        padded_rotations = torch.cat(
            [rotations, id.expand(*rotations.shape[:2], 1, 3, 2)], dim=-3
        )
        rot = torch.index_select(padded_rotations, dim=2, index=self.I)
        return {
            "positions": pos,
            "rotations": rot,
        }
        # NOTE: currently assumes rot is 6d and joints are at dim=2
