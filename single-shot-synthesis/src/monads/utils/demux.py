import typing

import torch

__all__ = [
    "DeMux",
]


class DeMux(torch.nn.Module):
    def __init__(
        self,
        joints: int,
        features: int,
        contacts: int = 4,
    ) -> None:
        super().__init__()
        self.joints = joints
        self.features = features
        self.ncontacts = contacts

    def forward(
        self, motion_data: torch.Tensor  # B, TEMPORAL, SPATIAL
    ) -> typing.Dict[str, torch.Tensor]:
        b = motion_data.shape[0]
        joint_rotations_sixd_flat = motion_data[:, :, : self.joints * self.features]
        joint_rotations_sixd = joint_rotations_sixd_flat.view(b, -1, self.joints, 3, 2)
        root_position = motion_data[:, :, -6:-3]
        contacts = motion_data[:, :, self.joints * self.features : -6].view(
            b, -1, self.ncontacts, 6
        )
        return {
            "joint_sixd": joint_rotations_sixd,
            "root_position": root_position,
            "contact_labels": contacts[:, :, :, :1],
            "contact_feats": contacts,
        }
