import logging
import typing

import colour
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

log = logging.getLogger(__name__)

try:
    import rerun as rr
except:
    log.error(
        f"Please `pip install rerun-sdk (0.20.1)` to use ReRun visualisation module."
    )

__all__ = ["skeleton3d", "mesh3d"]


def mesh3d(
    vertices: np.ndarray,
    faces: np.ndarray,
    path: str,
    color: str,
    input_style: np.ndarray,  # T, J, 3
    transferred_style: np.ndarray,  # T, J, 3
    optimization_step: typing.Optional[int] = None,
    lightning_step: typing.Optional[int] = None,
    iteration: typing.Optional[int] = None,
    log_seperate_frames: bool = False,
    style_labels: typing.List[str] = None,
) -> None:
    if optimization_step is not None:
        rr.set_time_sequence("optimization_step", optimization_step)
    elif lightning_step is not None:
        rr.set_time_sequence("lightning_step", lightning_step)
    elif iteration is not None:
        rr.set_time_sequence("iteration", iteration)
    color = colour.Color(color)

    def get_style_name(style_vector):
        if style_vector.sum() == 0.0:
            style = "neutral"
        else:
            style = style_labels[(style_vector == 1).nonzero()[0].item()]
        return style

    if len(vertices.shape) == 3:
        num_frames, _, __ = vertices.shape
        if num_frames != faces.shape[0]:
            faces = np.repeat(np.expand_dims(faces, 0), num_frames, axis=0)
        for fr in range(num_frames):
            rr.set_time_sequence("frame_nr", fr)
            o3d_mesh = trimesh.Trimesh(vertices=vertices[fr], faces=faces[fr])
            o3d_mesh.fix_normals()
            rr.log(
                path + f"/frame_nr{fr}" if log_seperate_frames else path,
                rr.Mesh3D(
                    vertex_positions=vertices[fr],
                    triangle_indices=faces[fr],
                    vertex_colors=np.tile(
                        np.array(color.get_rgb() + (1,)), (vertices.shape[1], 1)
                    ),
                    vertex_normals=np.array(o3d_mesh.vertex_normals),
                ),
            )
    elif len(vertices.shape) == 4:
        B, num_frames, _, __ = vertices.shape
        if num_frames != faces.shape[1]:
            faces = np.repeat(np.expand_dims(faces, 1), num_frames, axis=1)
        for b_ in range(B):
            for fr in range(num_frames):
                in_style = get_style_name(input_style[b_][fr])
                tr_style = get_style_name(transferred_style[b_][fr])
                rr.set_time_sequence("frame_mesh", fr)
                o3d_mesh = trimesh.Trimesh(
                    vertices=vertices[b_, fr], faces=faces[b_, fr]
                )
                o3d_mesh.fix_normals()
                rr.log(
                    f"/{path}/input_style/{in_style}/transferred_style/{tr_style}",
                    rr.Mesh3D(
                        vertex_positions=vertices[b_, fr],
                        triangle_indices=faces[b_, fr],
                        vertex_colors=np.tile(
                            np.array(color.get_rgb() + (1,)), (vertices.shape[1], 1)
                        ),
                        vertex_normals=np.array(o3d_mesh.vertex_normals),
                    ),
                )
    else:
        raise ValueError("Vertices must be 3D or 4D.")


def skeleton3d(
    kpts: np.ndarray,  # T, J, 3
    input_style: np.ndarray,  # T, J, 3
    transferred_style: np.ndarray,  # T, J, 3
    path: str,
    color: str = "yellow",
    parents: typing.List[int] = None,
    style_labels: typing.List[str] = None,
) -> None:
    color = colour.Color(color)
    # create topology
    tree = []

    def get_style_name(style_vector):
        if style_vector.sum() == 0.0:
            style = "neutral"
        else:
            style = style_labels[(style_vector == 1).nonzero()[0].item()]
        return style

    for i, parent in enumerate(parents):
        if parent != -1:
            tree.append([i, parent])
    for fr in range(kpts.shape[0]):
        edges = []
        for i, parent in tree:
            edges.append([kpts[fr][i], kpts[fr][parent]])
        in_style = get_style_name(input_style[fr])
        tr_style = get_style_name(transferred_style[fr])
        rr.set_time_sequence("frame_nr", fr)
        rr.log(
            f"/{path}/input_style/{in_style}/transferred_style/{tr_style}",
            rr.LineStrips3D(
                edges,
                colors=color.get_rgb(),
            ),
        )
        rr.log(
            f"/{path}/input_style/{in_style}/transferred_style/{tr_style}",
            rr.Points3D(
                kpts[fr],
                keypoint_ids=range(kpts[fr].shape[0]),
                colors=color.get_rgb(),
            ),
        )
