import logging
import os
import typing

import torch
from moai.utils.arguments import ensure_path

from src.exporters.bvh_exporting.bvh_utils import save

log = logging.getLogger(__name__)

__all__ = ["bvh_export"]


def bvh_export(
    tensors: typing.Dict[str, torch.Tensor],
    output_name: typing.List[str],
    export_path: typing.List[str],
    parents: typing.Union[str, typing.Sequence[str]],  # [B, J]
    position: typing.Union[str, typing.Sequence[str]],  # [B, J, 3]
    rotations: typing.Union[str, typing.Sequence[str]],  # [B, T, J, 3] (euler/degrees)
    offsets: typing.Union[str, typing.Sequence[str]],  # [B, T, 3]
    names: typing.Sequence[str],
    scale: typing.List[float] = 1.0,
    fps: typing.List[int] = 30,
    lightning_step: typing.Optional[int] = None,
    batch_idx: typing.Optional[int] = None,
) -> None:
    folder = ensure_path(log, "output folder", export_path)
    parents = [parents] if isinstance(parents, str) else list(parents)
    position = [position] if isinstance(position, str) else list(position)
    rotations = [rotations] if isinstance(rotations, str) else list(rotations)
    offsets = [offsets] if isinstance(offsets, str) else list(offsets)
    timestep = 1.0 / fps

    B, T = tensors[position].shape[:2]
    for b in range(B):
        out_filename = os.path.join(folder, f"{output_name}_{lightning_step}.bvh")
        save(
            out_filename,
            T,
            timestep,
            names,
            tensors[parents][b].detach().cpu().numpy(),
            (tensors[offsets][b] * scale).detach().cpu().numpy(),
            tensors[rotations][b].detach().cpu().numpy(),
            (tensors[position][b] * scale).detach().cpu().numpy(),
        )
