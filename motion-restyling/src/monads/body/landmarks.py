import logging
import os
import re
import typing
from collections import OrderedDict

import numpy as np
import toolz
import torch
from moai.utils.arguments.path import assert_path
from omegaconf import OmegaConf
from src.monads.body.models import ModelAttribute, ModelType

__escaped_glob_tokens_to_re__ = OrderedDict(
    (
        # Order of ``**/`` and ``/**`` in RE tokenization pattern doesn't matter because ``**/`` will be caught first no matter what, making ``/**`` the only option later on.
        # W/o leading or trailing ``/`` two consecutive asterisks will be treated as literals.
        (
            "/\*\*",
            "(?:/.+?)*",
        ),  # Edge-case #1. Catches recursive globs in the middle of path. Requires edge case #2 handled after this case.
        (
            "\*\*/",
            "(?:^.+?/)*",
        ),  # Edge-case #2. Catches recursive globs at the start of path. Requires edge case #1 handled before this case. ``^`` is used to ensure proper location for ``**/``.
        (
            "\*",
            "[^/]*",
        ),  # ``[^/]*`` is used to ensure that ``*`` won't match subdirs, as with naive ``.*?`` solution.
        ("\?", "."),
        ("\[\*\]", "\*"),  # Escaped special glob character.
        ("\[\?\]", "\?"),  # Escaped special glob character.
        (
            "\[!",
            "[^",
        ),  # Requires ordered dict, so that ``\[!`` preceded ``\[`` in RE pattern. Needed mostly to differentiate between ``!`` used within character class ``[]`` and outside of it, to avoid faulty conversion.
        ("\[", "["),
        ("\]", "]"),
    )
)

__escaped_glob_replacement__ = re.compile(
    "(%s)" % "|".join(__escaped_glob_tokens_to_re__).replace("\\", "\\\\\\")
)


def __glob_to_re__(pattern):
    return __escaped_glob_replacement__.sub(
        lambda match: __escaped_glob_tokens_to_re__[match.group(0)], re.escape(pattern)
    )


__all__ = [
    "LandmarksRegressor",
    "LandmarksRegressorData",
    "LandmarksOffsets",
    "MultiLandmarksRegressor",
]

log = logging.getLogger(__name__)


class LandmarksRegressorDense(torch.nn.Module):
    def __init__(
        self,
        data_file: str,
        landmarks: typing.Sequence[str],
    ):
        super().__init__()
        assert_path(log, "landmarks file", data_file)
        data = np.load(data_file)
        selected = []
        self.keys = []
        for landmark in landmarks:
            pattern = __glob_to_re__(landmark)
            matches = [re.fullmatch(pattern, file) for file in data.files]
            matches = [m for m in matches if bool(m)]
            if matches:
                for m in matches:
                    selected.append(data[m.string])
                    self.keys.append(m.string)
            else:
                log.warning(
                    f"Ignoring landmark {landmark} as it does not exist in the data file ({data_file})."
                )
        vert_counts = list(toolz.unique(map(lambda a: a.shape[-1], selected)))
        if len(vert_counts) > 1:
            log.warning(
                f"Mismatching vertex count landmark estimators loaded, ignoring all apart from those using {vert_counts[0]} vertices."
            )
            selected = toolz.remove(lambda a: a.shape[-1] != vert_counts[0], selected)
        self.register_buffer(
            "regressor",
            torch.stack(list(map(torch.from_numpy, selected))).float(),
            persistent=True,
        )

    def forward(
        self,
        vertices: torch.Tensor,
    ) -> torch.Tensor:
        self.regressor = self.regressor.to(vertices.dtype)
        if vertices.dim() == 2:
            return torch.einsum("vc,jv->jc", vertices, self.regressor)
        if vertices.dim() == 4:
            return torch.einsum("btvc,jv->btjc", vertices, self.regressor)
        return torch.einsum("bvc,jv->bjc", vertices, self.regressor)


LandmarksRegressor = LandmarksRegressorDense


class MultiLandmarksRegressor(torch.nn.Module):
    def __init__(self, config: str = None):
        super().__init__()
        self.config = config
        self.init_regressors()

    def load_config_from_yaml_file(
        self, config_path: str, key_in_config: str = "landmark_regressor_dynamic_specs"
    ):
        if not os.path.exists(config_path):
            log.error(
                f"Could not find the MultiLandmarksRegressor configuration file [{config_path}]."
            )
        return OmegaConf.load(config_path)[key_in_config]

    def init_regressors(self):
        """
        initialize landmark regressors from specified configuration
        """
        self.regressors = {}
        for key_str in self.config.keys():
            self.regressors[key_str] = LandmarksRegressor(**self.config[key_str])
            # self.modules[key_str].to(device)
        # for k, v in models.items():

    def get_module(self, key, device):
        key_str = f"{key[0]}_{key[1]}"
        if key_str not in self.config.keys():
            raise RuntimeError(f"Model type and gender requested not found{key}")

        self.regressors[key_str].to(device)
        return self.regressors[key_str]

    def forward(
        self,
        vertices: torch.Tensor,
        model_type: typing.Sequence[str],
        gender: typing.Sequence[str],
    ) -> torch.Tensor:

        key = (
            ModelType.to_str(model_type.flatten()[0].item()),
            ModelAttribute.to_str(gender.flatten()[0].item()),
        )
        module = self.get_module(key, vertices.device)
        return module(vertices)


class LandmarksRegressorData(LandmarksRegressorDense):
    def __init__(
        self,
        data_file: str,
        landmarks: typing.Sequence[str],
    ):
        super().__init__(data_file, landmarks)
        # the following assumes no landmarks were ignored !
        for landmark, regressor in zip(landmarks, self.regressor):
            self.register_buffer(landmark, regressor.clone(), persistent=True)

    def forward(
        self,
        void: torch.Tensor,
    ) -> torch.Tensor:
        return dict(self.named_buffers())


class LandmarksOffsets(torch.nn.Module):
    def __init__(
        self,
        parents: typing.Optional[typing.Sequence[int]] = None,
        preserve_root: typing.Optional[bool] = False,
    ):
        super().__init__()
        self.parents = parents
        self.preserve_root = preserve_root

    def forward(
        self,
        positions: torch.Tensor,
        parents: typing.Optional[
            torch.Tensor
        ] = None,  # NOTE: assumes parents are given in a landmark corresponding order
    ) -> torch.Tensor:
        if parents is not None:
            offsets = (
                positions - positions[:, parents]
                if len(positions.shape) == 3
                else positions - positions[:, :, parents]
            )
        else:
            offsets = positions - positions[:, self.parents]
        if self.preserve_root:
            if len(positions.shape) == 3:
                offsets[:, 0, :] = positions[:, 0, :]
            else:
                offsets[:, :, 0, :] = positions[:, :, 0, :]
        return offsets
