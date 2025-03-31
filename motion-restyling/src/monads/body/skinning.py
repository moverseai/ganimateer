import logging

import torch

__all__ = ["ApplyInverseBind", "Skinning"]

log = logging.getLogger(__name__)


class ApplyInverseBind(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        shaped_joints: torch.Tensor,  # [B, J, 3, 1]
        bone_transforms: torch.Tensor,  # [B, J, 4, 4]
    ) -> torch.Tensor:
        transformed_joints = bone_transforms[..., :3, :3] @ shaped_joints
        skinning_transforms = bone_transforms.clone()
        skinning_transforms[..., :3, 3] -= transformed_joints[..., 0]
        return skinning_transforms


class Skinning(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        vertices: torch.Tensor,  # (batch_size, vertex_count, 3)
        lbs_weights: torch.Tensor,  # (batch_size, vertex_count, joint_count)
        skinning_transforms: torch.Tensor,  # (batch_size, joint_count, 4, 4)   The relative (with respect to the root joint) rigid transformations for all the joints
    ) -> torch.Tensor:
        if len(vertices.shape) == 3:
            batch_size = skinning_transforms.shape[0]
            num_joints = skinning_transforms.shape[1]
            W = lbs_weights.expand([batch_size, -1, -1])
            V = vertices.expand([batch_size, -1, -1])
            T = torch.matmul(
                W, skinning_transforms.view(batch_size, num_joints, 16)
            ).view(batch_size, -1, 4, 4)
            homogen_coord = torch.ones(
                [batch_size, vertices.shape[1], 1],
                dtype=vertices.dtype,
                device=vertices.device,
            )
            v_posed_homo = torch.cat([V, homogen_coord], dim=2)
            v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))
            verts = v_homo[:, :, :3, 0]
            return verts
        elif len(vertices.shape) == 4:
            B = skinning_transforms.shape[0]
            verts = torch.zeros_like(vertices)
            for i in range(B):
                batch_size = skinning_transforms[i].shape[0]
                num_joints = skinning_transforms[i].shape[1]
                W = lbs_weights.expand([batch_size, -1, -1])
                V = vertices[i].expand([batch_size, -1, -1])
                T = torch.matmul(
                    W, skinning_transforms[i].view(batch_size, num_joints, 16)
                ).view(batch_size, -1, 4, 4)
                homogen_coord = torch.ones(
                    [batch_size, vertices[i].shape[1], 1],
                    dtype=vertices.dtype,
                    device=vertices.device,
                )
                v_posed_homo = torch.cat([V, homogen_coord], dim=2)
                v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))
                verts[i] = v_homo[:, :, :3, 0]
            return verts
