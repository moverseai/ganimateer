import typing

import torch
from moai.monads.sampling.torch.interpolate import Interpolate


class PyramidLengths(torch.nn.Module):
    def __init__(
        self,
        ratio: float,
        scale: float,
        mode: str = "linear",  # one of ['nearest', 'linear', 'bilinear', 'area', 'bicubic', 'trilinear']
        align_corners: bool = False,
        recompute_scale_factor: bool = False,
        preserve_aspect_ratio: bool = False,
    ) -> None:
        super().__init__()
        self.ratio = ratio
        self.scale = scale
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor
        self.preserve_aspect_ratio = preserve_aspect_ratio

    def forward(self, tensor: torch.Tensor) -> typing.Dict[str, torch.Tensor]:
        total_length = tensor.shape[-1]
        lengths = []
        lengths.append(int(total_length * self.ratio))
        while lengths[-1] < total_length:
            lengths.append(int(lengths[-1] * self.scale))
            if lengths[-1] == lengths[-2]:
                lengths[-1] += 1
        lengths[-1] = total_length
        out = {}
        for i, length in enumerate(lengths):
            out[f"level_{i}"] = Interpolate(
                width=length,
                mode=self.mode,
                align_corners=self.align_corners,
                height=1,
                recompute_scale_factor=self.recompute_scale_factor,
                preserve_aspect_ratio=self.preserve_aspect_ratio,
            ).forward(tensor)
        return out
