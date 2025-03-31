import logging

import torch

__all__ = ["BlendShapes"]

log = logging.getLogger(__name__)


class BlendShapes(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        template: torch.Tensor,
        coefficients: torch.Tensor,
        blendshapes: torch.Tensor,
    ) -> torch.Tensor:
        if len(coefficients.shape) == 4:
            offsets = torch.einsum(
                "bl,mkl->bmk",
                [coefficients.view(coefficients.shape[0], -1), blendshapes],
            )
        elif len(coefficients.shape) == 5:
            offsets = torch.zeros_like(template)
            for i in range(coefficients.shape[0]):
                offsets[i] = torch.einsum(
                    "bl,mkl->bmk",
                    [coefficients[i].view(coefficients[i].shape[0], -1), blendshapes],
                )

        elif len(coefficients.shape) == 3:
            B, T, C = coefficients.shape
            s = C
            offsets = torch.einsum("btc,mkl->bmt", [coefficients, blendshapes[..., :s]])
            offsets = offsets.view(B, T, -1).unsqueeze(-1)
        else:
            s = coefficients.shape[1]
            offsets = torch.einsum("bl,mkl->bmk", [coefficients, blendshapes[..., :s]])
        return template + offsets
