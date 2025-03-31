import logging

import omegaconf.omegaconf
import torch

log = logging.getLogger(__name__)


__all__ = ["NeutralTransfer"]


class NeutralTransfer(torch.nn.Module):
    def __init__(
        self,
        configuration: omegaconf.DictConfig,
    ) -> None:
        super().__init__()
        # create the correct num of lstm neutral layers
        self.neutral_branch = torch.nn.LSTM(
            input_size=configuration.in_features,
            hidden_size=configuration.hidden_features,
            num_layers=configuration.blocks,
            batch_first=configuration.batch_first,
        )
        self.init_hidden_state = torch.nn.Parameter(
            torch.zeros(
                configuration.blocks,
                configuration.content_num,
                configuration.in_features,
            )
        )
        self.init_cell_state = torch.nn.Parameter(
            torch.zeros(
                configuration.blocks,
                configuration.content_num,
                configuration.in_features,
            )
        )
        log.info(
            f"Neutral Branch initialised with total layers: {configuration.blocks} and content num: {configuration.content_num}"
        )

    def forward(self, input: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        batch_size, T, _ = input.shape
        content_index = torch.argmax(
            content.reshape(batch_size, T, -1).mean(dim=1), dim=-1
        )
        h0_neutral = self.init_hidden_state[:, content_index, :]
        c0_neutral = self.init_cell_state[:, content_index, :]
        neutral_output, (hn, cn) = self.neutral_branch(input, (h0_neutral, c0_neutral))
        return neutral_output
