import typing

import numpy as np
import torch


class CreateRandomClips(torch.nn.Module):
    def __init__(self, window_size: int) -> None:
        super(CreateRandomClips, self).__init__()
        self.window_size = window_size

    def forward(self, motion: torch.Tensor) -> typing.Dict[str, torch.Tensor]:
        max_offset = motion.shape[1] - self.window_size
        num_clips = motion.shape[1] // self.window_size
        clips = []
        for i in range(num_clips):
            offsets = np.random.randint(max_offset, size=2)
            clips.append(motion[0, offsets[0] : offsets[0] + self.window_size, :])
        return torch.stack(clips, dim=0)
