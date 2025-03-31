import typing

import numpy as np
import torch

__all__ = ["FootContact"]


class FootContact(torch.nn.Module):
    def __init__(
        self,
        threshold: float,
        joint_indices: typing.Sequence[int],
        zero_pad: int = 0,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.indices = joint_indices
        self.zero_pad = zero_pad

    def forward(self, velocity: torch.Tensor) -> torch.Tensor:
        if len(velocity.shape) == 4:
            # assumes [B, T, J, 3]
            contact = velocity[:, :, self.indices, :].sum(axis=-1) < self.threshold
            labels = contact.int()
            return labels

        contact = velocity[..., self.indices] < self.threshold
        labels = contact.int()
        pad = torch.zeros_like(
            labels[:1, :] if velocity.ndim <= 2 else labels[:, :1, :]
        )
        contact_labels = torch.cat([pad, contact], dim=0 if velocity.ndim <= 2 else 1)
        out = {"raw": contact_labels}
        if self.zero_pad:
            zeros = torch.zeros(
                *contact_labels.shape, self.zero_pad, device=contact_labels.device
            )
            contact_labels_padded = torch.cat(
                [contact_labels[..., np.newaxis], zeros], dim=-1
            )
            out["padded"] = contact_labels_padded
        return out
