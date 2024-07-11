import typing
from dataclasses import dataclass
from io import TextIOWrapper

import numpy as np

__all__ = ["save"]


@dataclass
class BvhJoint:
    id: int
    children: list


def _add_to_parent(pid: int, cid: int, joint: BvhJoint) -> bool:
    if joint.id == pid:
        joint.children.append(BvhJoint(cid, []))
        return True
    if not joint.children:
        return False
    else:
        for c in joint.children:
            if _add_to_parent(pid, cid, c):
                return True
        return False


def _create_hierarchy(parents: np.ndarray) -> BvhJoint:
    root = BvhJoint(0, [])  # NOTE: assumes root is id=0
    checklist = np.zeros_like(parents)
    checklist[0] = 1
    while checklist.sum() < len(parents):
        for i, p in enumerate(parents[1:], start=1):
            if checklist[i]:
                continue
            if _add_to_parent(p, i, root):
                checklist[i] = 1
    return root


def _write_joint(
    file: TextIOWrapper,
    joint: BvhJoint,
    ordered_ids: typing.Sequence[int],
    names: typing.Sequence[str],
    offsets: np.ndarray,
    indent: int,
    precision: int,
) -> None:
    isFirst = joint.id == 0
    offset = offsets[joint.id]
    name = names[joint.id]
    ordered_ids.append(joint.id)
    tab = "\t"
    file.write(f'{tab*indent}{"ROOT" if isFirst else "JOINT"} {name}\n')
    file.write(f"{tab*indent}{{\n")
    file.write(f"{tab*(indent+1)}OFFSET ")
    file.write(f"{round(offset[0], precision)} ")
    file.write(f"{round(offset[1], precision)} ")
    file.write(f"{round(offset[2], precision)}\n")
    if isFirst:
        file.write(
            f"{tab*(indent+1)}CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation \n"
        )
    else:
        file.write(f"{tab*(indent+1)}CHANNELS 3 Xrotation Yrotation Zrotation \n")
    if len(joint.children) > 0:
        for child in joint.children:
            _write_joint(
                file, child, ordered_ids, names, offsets, indent + 1, precision
            )
    else:
        file.write(f"{tab*(indent+1)}End Site\n")
        file.write(f"{tab*(indent+1)}{{\n")
        file.write(f"{tab*(indent+2)}OFFSET ")
        file.write(f"{round(offset[0], precision)} ")
        file.write(f"{round(offset[1], precision)} ")
        file.write(f"{round(offset[2], precision)}\n")
        file.write(f"{tab*(indent+1)}}}\n")
    file.write(f"{tab*indent}}}\n")


def _write_motion(
    file: TextIOWrapper, rotations: np.ndarray, position: np.ndarray, prec: int
) -> None:
    pass
    for pos, rots in zip(position, rotations):
        file.write(
            f"{round(pos[0], prec)} {round(pos[1], prec)} {round(pos[2], prec)} "
        )
        for rot in rots:
            file.write(
                f"{round(rot[0], prec)} {round(rot[1], prec)} {round(rot[2], prec)} "
            )
        file.write("\n")


def save(
    path: str,
    frames: int,
    timestep: float,
    names: typing.Sequence[str],
    parents: np.ndarray,
    offsets: np.ndarray,
    rotations: np.ndarray,
    position: np.ndarray,
    precision: int = 6,
) -> None:
    root = _create_hierarchy(parents)
    with open(path, "w") as file:
        file.write("HIERARCHY\n")
        ordered_ids = []
        _write_joint(file, root, ordered_ids, names, offsets, 0, precision)
        file.write("MOTION\n")
        file.write(f"Frames: {frames}\n")
        file.write(f"Frame Time: {timestep}\n")
        _write_motion(file, rotations[:, ordered_ids, ...], position, precision)
        file.write("\n")
