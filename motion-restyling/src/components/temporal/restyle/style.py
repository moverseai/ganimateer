import logging

import omegaconf.omegaconf
import torch

log = logging.getLogger(__name__)


__all__ = ["StyleTransfer"]


class StyleTransfer(torch.nn.Module):
    def __init__(
        self,
        configuration: omegaconf.DictConfig,
    ) -> None:
        super().__init__()
        # create the correct num of lstm style layers
        self.style_layers = torch.nn.ModuleList(
            [
                torch.nn.LSTM(
                    input_size=configuration.in_features,
                    hidden_size=configuration.hidden_features,
                    num_layers=configuration.blocks,
                    batch_first=configuration.batch_first,
                )
                for _ in range(configuration.num_styles)
            ]
        )
        # TODO: add a random init for the hidden state
        self.init_hidden_state = torch.nn.Parameter(
            torch.zeros(
                configuration.blocks,
                configuration.num_styles,
                configuration.in_features,
            )
        )
        self.init_cell_state = torch.nn.Parameter(
            torch.zeros(
                configuration.blocks,
                configuration.num_styles,
                configuration.in_features,  # TODO: take this from in_feaures or hidden_features
            )
        )
        log.info(
            f"Style Branch initialised with total layers: {configuration.blocks} and num_styles: {configuration.num_styles}"
        )

    def forward(
        self, input: torch.Tensor, transferred_style: torch.Tensor
    ) -> torch.Tensor:
        ra_result = []
        batch_size, T, _ = input.shape
        for i, branch in enumerate(self.style_layers):
            h0_ra = (
                self.init_hidden_state[:, i, :].unsqueeze(1).repeat(1, batch_size, 1)
            )
            c0_ra = self.init_cell_state[:, i, :].unsqueeze(1).repeat(1, batch_size, 1)
            ra_result.append(branch(input, (h0_ra, c0_ra))[0])
        ra_result = torch.stack(ra_result) * transferred_style.permute(
            2, 0, 1
        ).unsqueeze(-1)
        return ra_result.sum(dim=0)  # caclulate the sum of the style features
