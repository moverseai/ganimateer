import logging
import typing

import numpy as np

log = logging.getLogger(__name__)

# from collections.abc import Callable

try:
    import rerun as rr
except:
    log.error(f"Please `pip install rerun-sdk` to use ReRun visualisation module.")

__all__ = ["skeleton3d", "multi_skeleton3d"]


def skeleton3d(
    kpts: np.ndarray,
    path: str,
    skeleton_type: str = "mixamo",
    batch_item: int = 0,
    lightning_step: typing.Optional[int] = None,
) -> None:
    step = lightning_step // 7
    for j in range(kpts.shape[1]):
        rr.set_time_sequence("frame_nr", j)
        p = f"{path}/{step}"
        rr.log(
            p,
            rr.Points3D(
                kpts[batch_item][j],
                labels=skeleton_type,
                keypoint_ids=range(kpts[batch_item][j].shape[0]),
            ),
        )


def multi_skeleton3d(
    kpts: np.ndarray,
    path: str,
    skeleton_type: str = "mixamo",
    batch_idx: typing.Optional[int] = None,
) -> None:
    for j in range(kpts.shape[1]):
        rr.set_time_sequence("frame_nr", j)
        p = f"{path}/{batch_idx}"
        rr.log(
            p,
            rr.Points3D(
                kpts[0][j],
                labels=skeleton_type,
                keypoint_ids=range(kpts[0][j].shape[0]),
            ),
        )
