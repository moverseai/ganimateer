import torch

__all__ = ["VeloLabelConsistencyLoss"]


class VeloLabelConsistencyLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,  # demuxed foot contact positions
        gt: torch.Tensor,  # velocities
    ) -> torch.Tensor:
        contact = torch.sigmoid((pred.squeeze(-1) - 0.5) * 2 * 6)  # .detach()
        # contact = 1 / (1 + torch.exp(5-10*pred.squeeze(-1)))
        return torch.mean(gt * contact)
