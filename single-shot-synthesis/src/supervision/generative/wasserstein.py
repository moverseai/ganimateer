import torch

# NOTE: adapted from: https://github.com/watml/robust-gan/blob/master/loss.py#L91

__all__ = [
    "GeneratorWasserstein",
    "DiscriminatorWasserstein",
]


class GeneratorWasserstein(torch.nn.Module):
    def __init__(
        self,
        per_batch: bool = False,
    ):
        super().__init__()
        self.per_batch = per_batch

    def forward(self, fake: torch.Tensor) -> torch.Tensor:
        fdims = list(range(1 if self.per_batch else 0, len(fake.shape)))
        return -torch.mean(fake, dim=fdims)


class DiscriminatorWasserstein(torch.nn.Module):
    def __init__(
        self,
        per_batch: bool = False,
    ):
        super().__init__()
        self.per_batch = per_batch

    def forward(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        rdims = list(range(1 if self.per_batch else 0, len(real.shape)))
        fdims = list(range(1 if self.per_batch else 0, len(fake.shape)))
        return -torch.mean(real, dim=rdims) + torch.mean(fake, dim=fdims)
