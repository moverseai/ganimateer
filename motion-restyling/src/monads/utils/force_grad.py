import torch


class ForceGrad(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        tensor.requires_grad_(True)  # TODO: fix this hack !
        return tensor
