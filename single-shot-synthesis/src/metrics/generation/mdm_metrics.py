import typing

import numpy as np
import torch
from moai.validation.metric import MoaiMetric

__all__ = ["MDMInterDiversity", "MDMIntraDiversity"]


class MDMInterDiversity(MoaiMetric):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(pred - gt)

    def compute(
        self,
        diversities: typing.Sequence[np.ndarray],
    ) -> typing.Union[np.ndarray, typing.Mapping[str, np.ndarray]]:
        return sum(diversities) / len(diversities)


class MDMIntraDiversity(MoaiMetric):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
    ) -> torch.Tensor:
        selected = np.random.choice(pred.shape[0], 2)
        pred_intra_div = torch.linalg.norm(pred[selected[0]] - pred[selected[1]])
        gt_intra_div = torch.linalg.norm(gt[selected[0]] - gt[selected[1]])
        return torch.abs(pred_intra_div - gt_intra_div)

    def compute(
        self,
        diversities: typing.Sequence[np.ndarray],
    ) -> typing.Union[np.ndarray, typing.Mapping[str, np.ndarray]]:
        return sum(diversities) / len(diversities)
    