import logging
import typing
from operator import itemgetter

import numpy as np
import torch
from moai.utils.arguments import assert_path

from src.data.load_bvh import load

log = logging.getLogger(__name__)

__all__ = ["BioVisionHierarchyFile"]


class BioVisionHierarchyFile(torch.utils.data.Dataset):
    def __init__(
        self,
        filename: str,
        subset: typing.Optional[typing.Sequence[int]] = None,
        scale: float = 1.0,
        fps: int = 30,
    ) -> None:
        super().__init__()
        self.files_count = 0
        assert_path(log, "BVH file", filename)
        pos, euler, off, self.parents, self.joint_names, self.timestep = load(filename)
        timescale = round(1.0 / fps / self.timestep)
        pos = pos[::timescale]
        euler = euler[::timescale]
        self.positions = pos.astype(np.float32) * scale
        self.joint_offsets = off.astype(np.float32) * scale
        T, J, _ = euler.shape
        # rotations = R.from_euler('XYZ', euler.reshape(T * J, 3), degrees=True)
        # self.rotations = rotations.as_matrix().reshape(T, J, 3, 3).astype(np.float32)
        self.rotations = euler.astype(np.float32)
        if subset:  # should include '0' as the first (root) element
            original_names = self.joint_names
            self.joint_names = list(itemgetter(*subset)(self.joint_names))
            parents_subset = itemgetter(*subset)(self.parents)
            self.parents = [-1] + [
                self.joint_names.index(original_names[p]) for p in parents_subset[1:]
            ]
            self.joint_offsets = self.joint_offsets[subset]
            self.rotations = self.rotations[:, subset, ...]
        self.parents = np.array(self.parents, dtype=np.int64)
        log.info(f"Loaded {len(self)} (BVH) motion samples from {filename}.")

    def __len__(self) -> int:
        return 1  # single sequence/file

    def __getitem__(self, _: int) -> typing.Dict[str, torch.Tensor]:
        return {
            "joint_offsets": self.joint_offsets,
            "root_position": self.positions,
            "joint_rotations": self.rotations,
            "joint_parents": self.parents,
        }
