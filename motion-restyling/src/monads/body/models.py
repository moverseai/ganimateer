import logging
import os
import pickle
import typing
from enum import IntEnum

import numpy as np
import toolz
import torch
import trimesh
from scipy.sparse import issparse

log = logging.getLogger(__name__)

_SMPL_RENAMES_ = {
    "J_regressor": "kintree_regressor",
    "shapedirs": "blendshapes",
    "posedirs": "pose_correctives",
    "weights": "skinning_weights",
    "v_template": "vertices",
    "f": "faces",
    "kintree_table": "parents",
}


class ModelAttribute(IntEnum):
    MALE = 0
    FEMALE = 1
    NEUTRAL = 2

    @classmethod
    def get_enum(cls, value):
        return ModelAttribute(cls.to_int(value))

    @classmethod
    def names(cls):
        return ["male", "female", "neutral"]

    @classmethod
    def to_str(cls, value):
        return cls.names()[value - cls.MALE]

    @classmethod
    def to_int(cls, s):
        return cls.names().index(s) + cls.MALE


class ModelType(IntEnum):
    SMPL = 0

    @classmethod
    def get_enum(cls, value):
        return ModelType(cls.to_int(value))

    @classmethod
    def names(cls):
        return ["smpl"]

    @classmethod
    def to_str(cls, value):
        return cls.names()[value - cls.SMPL]

    @classmethod
    def to_int(cls, s):
        return cls.names().index(s) + cls.SMPL


def _pick(whitelist, d):
    return toolz.keyfilter(lambda k: k in whitelist, d)


def load_body_model(
    model_type: ModelType, filename: str, compute_normals: bool = False
) -> typing.Mapping[str, np.array]:
    data = _BODY_LOADERS_[model_type](filename)
    data["parents"] = data["parents"][0]
    data["parents"][0] = 0

    if compute_normals:
        mesh = trimesh.Trimesh(
            vertices=data["vertices"],
            faces=data["faces"].astype(np.int64),
            process=False,
        )
        data["normals"] = np.copy(mesh.vertex_normals.astype(np.float32))

    return data


def _convert_smpl_pkl_data(datum):
    if issparse(datum):
        array = datum.toarray()
    elif not isinstance(datum, (str, np.ndarray)):
        array = datum.x
    else:
        array = datum
    if isinstance(array, np.ndarray) and np.issubdtype(array.dtype, np.floating):
        return array.astype(np.float32)
    elif isinstance(array, np.ndarray) and np.issubdtype(array.dtype, np.integer):
        return array.astype(np.int64)
    else:
        return array


def _load_smpl_body_model(filename: str) -> typing.Mapping[str, np.array]:
    _, ext = os.path.splitext(filename)
    if ext == ".pkl":

        with open(filename, "rb") as f:
            data = toolz.valmap(
                _convert_smpl_pkl_data, pickle.load(f, encoding="latin1")
            )
    elif ext == ".npz":
        data = toolz.valmap(_convert_smpl_pkl_data, dict(np.load(filename)))
    else:
        raise RuntimeError(
            "The SMPL body model can only be loaded from *.pkl or *.npz files."
        )
    return toolz.keymap(lambda k: _SMPL_RENAMES_[k], _pick(_SMPL_RENAMES_.keys(), data))


_BODY_LOADERS_ = {
    ModelType.SMPL: _load_smpl_body_model,
}


class BodyModelData(torch.nn.Module):
    def __init__(
        self,
        body_type: str,
        filename: str,
        persistent: bool = True,
        compute_normals: bool = False,
    ):
        super().__init__()
        if not os.path.exists(filename):
            log.error(f"Could not find the body model file [{filename}].")
        data = toolz.valmap(
            lambda a: torch.from_numpy(a if a.dtype != np.uint else a.astype(np.int32)),
            load_body_model(
                ModelType.get_enum(body_type), filename, compute_normals=compute_normals
            ),
        )
        for k, v in data.items():
            self.register_buffer(k, v, persistent=persistent)

    def forward(
        self, void: typing.Optional[typing.Any] = None
    ) -> typing.Mapping[str, torch.Tensor]:
        return dict(self.named_buffers())


class MultiBodyModelData(torch.nn.ModuleDict):
    def __init__(
        self,
        config: str = None,
        persistent: bool = True,
    ):
        super().__init__()
        self.config = config
        self.persistent = persistent
        for k, v in self.config.items():
            self[k] = BodyModelData(**v, persistent=persistent)

    def forward(
        self,
        model_type: typing.List[torch.LongTensor],
        gender: typing.List[torch.LongTensor],
    ) -> typing.Mapping[str, torch.Tensor]:
        key = f"{ModelType.to_str(int(model_type.flatten()[0]))}_{ModelAttribute.to_str(int(gender.flatten()[0]))}"
        return self[key]()
