import logging

import kornia as kn
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

log = logging.getLogger(__name__)


class FaceZFwd(torch.nn.Module):
    def __init__(
        self, left_hip: int, right_hip: int, left_shoulder: int, right_shoulder: int
    ):
        """
        Rotate the input motion to z fwd.
        """
        super(FaceZFwd, self).__init__()
        self.left_hip = left_hip
        self.right_hip = right_hip
        self.left_shoulder = left_shoulder
        self.right_shoulder = right_shoulder

    def y_rotation_from_positions(
        self, left_hip, right_hip, left_shoulder, right_shoulder
    ):
        # Compute across vector (local x-axis) from hips and shoulders
        across = left_hip - right_hip + left_shoulder - right_shoulder
        across /= torch.linalg.norm(across, axis=-1, keepdims=True)

        # Compute forward vector as the cross product with the y-axis
        # forward = torch.cross(across, torch.Tensor([0, 1, 0]).to(across.device))
        forward = torch.cross(
            across,
            torch.Tensor([0, 1, 0])[np.newaxis, np.newaxis, ...].to(across.device),
        )
        # forward = gaussian_filter1d(forward, sigma=20, axis=0, mode="nearest")
        forward /= torch.linalg.norm(forward, axis=-1, keepdims=True)

        # Define the target forward vector (global z-axis)
        target_forward = torch.Tensor([0, 0, 1]).to(forward.device)
        # find in between rotation
        # a = torch.cross(forward, target_forward)
        a = torch.cross(forward, target_forward[np.newaxis, np.newaxis, ...])
        w = torch.sqrt(
            (torch.linalg.norm(forward) ** 2).sum(axis=-1)
            * (torch.linalg.norm(target_forward).sum(-1) ** 2)
        ) + (forward * target_forward[np.newaxis, np.newaxis, ...]).sum(axis=-1)
        q = torch.concatenate((w[..., np.newaxis], a), axis=-1)
        q /= torch.linalg.norm(q, axis=-1, keepdims=True)

        # Calculate the rotation matrix needed to align the forward vector with the target
        batch, frames = forward.shape[:2]
        rot_mat = np.zeros((batch, frames, 3, 3))
        for b in range(batch):
            for f in range(frames):
                rot_mat[b, f] = R.align_vectors(
                    target_forward.cpu(), forward[b, f].cpu()
                )[0].as_matrix()

        return torch.from_numpy(rot_mat).float().to(forward.device)

    def forward(
        self,
        rotations: torch.Tensor,  # (batch, frames, joints, 3, 3)
        positions: torch.Tensor,  # (batch, frames, joints, 3)
    ) -> torch.Tensor:
        left_hip = positions[:, :, self.left_hip]
        right_hip = positions[:, :, self.right_hip]
        left_shoulder = positions[:, :, self.left_shoulder]
        right_shoulder = positions[:, :, self.right_shoulder]
        rot_mat_to_apply = self.y_rotation_from_positions(
            left_hip, right_hip, left_shoulder, right_shoulder
        )
        new_joint_positions = torch.einsum(
            "bfij,bfvj->bfvi", rot_mat_to_apply, positions
        )
        new_root_rot = rot_mat_to_apply @ rotations[:, :, 0, :, :]
        new_rotations = torch.cat(
            [new_root_rot.unsqueeze(2), rotations[:, :, 1:, :, :]], dim=2
        )
        # convert to rotation rotvec
        new_rotations = kn.geometry.rotation_matrix_to_axis_angle(new_rotations)
        return {
            "positions": new_joint_positions,
            "rotations": new_rotations,
            "transformed_matrix": rot_mat_to_apply,
        }


class FixRotation(torch.nn.Module):
    def __init__(self):
        """
        Rotate the input motion to the original facing direction, rather than z fwd.
        """
        super(FixRotation, self).__init__()

    def forward(
        self,
        vertices: torch.Tensor,
        rotation_matrix: torch.Tensor,
        translation: torch.Tensor,
    ) -> torch.Tensor:
        rotated_vertices = torch.einsum(
            "bfij,bfvj->bfvi", rotation_matrix.transpose(3, 2), vertices
        )
        rotated_translation = torch.einsum(
            "bfij,bfkj->bfki", rotation_matrix.transpose(3, 2), translation
        )
        rotated_vertices = rotated_vertices + rotated_translation
        return rotated_vertices
