import logging

import torch

log = logging.getLogger(__name__)

__all__ = ["GradientPenalty"]


class GradientPenalty(torch.nn.Module):
    def __init__(self, mode: str = "mean") -> None:
        super(GradientPenalty, self).__init__()
        self.mode = mode

    def forward(
        self,
        inputs: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = scores.size(0)
        grad = torch.autograd.grad(
            outputs=scores.mean() if self.mode == "mean" else scores.sum(),
            inputs=inputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad = (grad**2).sum() / batch_size
        return grad
