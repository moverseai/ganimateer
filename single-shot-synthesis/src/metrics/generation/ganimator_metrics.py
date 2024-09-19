import logging
import typing

import numpy as np
import torch
from moai.validation.metric import MoaiMetric
from numba import jit

log = logging.getLogger(__name__)

__all__ = [
    "LocalDiversity",
    "GlobalDiversity",
    "Coverage",
]

try:
    from ganimator_eval_kernel import nn_dp as _py_nn_dp_kernel
    from ganimator_eval_kernel import prepare_group_cost as _py_prepare_group_cost
except:
    log.warning(
        "[yellow bold]:warning:[/] Could not load the [cyan underline]ganimator-eval-kernel[/] package, reverting to the [green underline]numba[/] implementation."
    )

    @jit(nopython=True)
    def _py_prepare_group_cost(group_cost, cost):
        L = cost.shape[0]
        L2 = cost.shape[1]

        for i in range(0, L):
            for j in range(i + 1, L + 1):
                k = 0
                while k + j - i - 1 < L2:
                    group_cost[i, j, k] = (
                        group_cost[i, j - 1, k] + cost[j - 1, k + j - i - 1]
                    )
                    k += 1
        return

    @jit(nopython=True)
    def _py_nn_dp_kernel(G, E, F, Cost, tmin, L, Nt):
        g = G
        e = E
        f = F
        cost = Cost
        g[0] = 0

        for i in range(tmin, L + 1):
            for k in range(0, Nt):
                l = 0
                while l <= i - tmin:
                    new_val = g[l] + cost[l, i, k]
                    l += 1
                    if new_val < g[i]:
                        g[i] = new_val
                        e[i] = k
                        f[i] = l


def _group_cost_from_tensors(src, tgt, use_pos=False):
    # cdist equivalent
    src_pos = src.unsqueeze(1)
    tgt_pos = tgt.unsqueeze(0)
    cost = torch.norm(src_pos - tgt_pos, dim=2)
    
    L = cost.shape[0]  # ground truth num_frames
    L_target = cost.shape[1]  # generated sequence number of frames

    group_cost = np.zeros((L, L + 1, L_target))
    group_cost.fill(np.inf)
    for i in range(L):
        group_cost[i, i] = 0
    cost = cost.cpu().numpy()
    _py_prepare_group_cost(group_cost, cost)

    return torch.from_numpy(group_cost).to(src.device)


def _group_cost_from_rotmat(src, tgt, use_pos=False):
    # cdist equivalent
    src_pos = src.unsqueeze(1)  # e.g. [T1, K, 3, 3]-> [T1 ,1, K, 3, 3]
    tgt_pos = tgt.unsqueeze(0)  # e.g. [T2, K, 3, 3] -> [1, T2, K, 3, 3]
    cost = torch.norm(src_pos - tgt_pos, dim=[-1, -2]).mean(dim=-1)  # [T1, T2]

    L = cost.shape[0]  # ground truth num_frames
    L_target = cost.shape[1]  # generated sequence number of frames

    group_cost = np.zeros((L, L + 1, L_target))
    group_cost.fill(np.inf)
    for i in range(L):
        group_cost[i, i] = 0
    cost = cost.cpu().numpy()
    _py_prepare_group_cost(group_cost, cost)

    return torch.from_numpy(group_cost).to(src.device)


def _nn_dp_fast(group_cost, tmin):
    L = group_cost.shape[0]
    L_target = group_cost.shape[-1]
    G = np.zeros((L + 5,), dtype=np.float64)
    G.fill(np.inf)
    E = np.zeros(G.shape, dtype=np.int32)
    F = np.zeros_like(E)
    label = np.zeros(L, dtype=np.int32)

    _py_nn_dp_kernel(G, E, F, group_cost, tmin, L, L_target)

    lengths = []
    seps = []
    p = L
    while p > 0:
        label[F[p] : p] = E[p] + np.arange(p - F[p])
        seps.append((E[p], E[p] + p - F[p]))
        lengths.append(p - F[p])
        p = F[p]

    return G[L], label


def _calc_perwindow_cost(group_cost, tmin, keepall=False):
    res = 0.0
    all_res = []
    for i in range(group_cost.shape[0] - tmin):
        cost = np.min(group_cost[i, i + tmin]) / tmin
        res += cost
        all_res.append(cost)
    if keepall:
        return res / (group_cost.shape[0] - tmin), all_res
    else:
        return res / (group_cost.shape[0] - tmin)


class Coverage(MoaiMetric):
    def __init__(
        self,
        tmin: int = 30,
        threshold: float = 2.0,
    ):
        super().__init__()
        self.tmin = tmin
        self.threshold = threshold

    def forward(
        self, pred: torch.Tensor, gt: torch.Tensor  # [B, T, ...]  # [B, T, ...]
    ) -> torch.Tensor:
        assert pred.shape[0] == 1 and gt.shape[0] == 1
        group_cost = _group_cost_from_tensors(pred[0], gt[0])  # ground truth, generated
        # iterate for each window
        res = []
        for i in range(group_cost.shape[0] - self.tmin):
            cost = torch.min(group_cost[i, i + self.tmin]) / self.tmin
            res.append(1.0 if cost < self.threshold else 0.0)
        return torch.mean(torch.Tensor([res]))

    def compute(
        self,
        coverages: typing.Sequence[np.ndarray],
    ) -> typing.Union[np.ndarray, typing.Mapping[str, np.ndarray]]:
        return sum(coverages) / len(coverages)


class GlobalDiversity(MoaiMetric):
    def __init__(
        self,
        tmin: int = 30,
    ):
        super().__init__()
        self.tmin = tmin

    def forward(
        self, pred: torch.Tensor, gt: torch.Tensor  # [B, T, ..]  # [B, T, ..]
    ) -> torch.Tensor:
        assert pred.shape[0] == 1 and gt.shape[0] == 1
        group_cost = _group_cost_from_tensors(pred[0], gt[0])
        val, label = _nn_dp_fast(group_cost.cpu().numpy(), self.tmin)
        res = val / label.shape[0]

        return torch.from_numpy(np.array(res)).to(pred.device)

    def compute(
        self,
        diversities: typing.Sequence[np.ndarray],
    ) -> typing.Union[np.ndarray, typing.Mapping[str, np.ndarray]]:
        return sum(diversities) / len(diversities)


class LocalDiversity(MoaiMetric):
    def __init__(self, tmin: int = 15, keepall: bool = False):
        super().__init__()
        self.tmin = tmin
        self.keepall = keepall

    def forward(
        self, pred: torch.Tensor, gt: torch.Tensor  # [B, T, ..]  # [B, T, ..]
    ) -> torch.Tensor:
        assert pred.shape[0] == 1 and gt.shape[0] == 1
        group_cost = _group_cost_from_tensors(pred[0], gt[0])
        res = _calc_perwindow_cost(group_cost.cpu().numpy(), self.tmin, self.keepall)

        return torch.from_numpy(np.array(res)).to(pred.device)

    def compute(
        self,
        diversities: typing.Sequence[np.ndarray],
    ) -> typing.Union[np.ndarray, typing.Mapping[str, np.ndarray]]:
        return sum(diversities) / len(diversities)
