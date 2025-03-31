import logging

import omegaconf.omegaconf
import torch

log = logging.getLogger(__name__)

__all__ = ["Discriminator"]


class Discriminator(torch.nn.Module):
    def __init__(
        self,
        configuration: omegaconf.DictConfig,
    ):
        super().__init__()
        layers = []
        current_size = (
            configuration.in_features
        )  # NOTE: should be the sum of the input vector (e..g. if position and velocity are concatenated, this should be the sum of their dimensions)
        dummy_data = torch.zeros(1, current_size, configuration.episode_length)
        for _ in range(configuration.blocks):
            layers.append(self.conv_layer(3, current_size, (current_size // 3) * 2))
            layers.append(torch.nn.LeakyReLU())
            current_size = (current_size // 3) * 2
        self.features = torch.nn.Sequential(*layers)
        self.last_layer = self.conv_layer(3, current_size, configuration.feature_dim)
        input_size = (
            configuration.attention_dim
        )  # NOTE: this should be the dimension of the input to the attention module
        self.attention_features = torch.nn.Sequential(
            torch.nn.Linear(input_size, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
        )
        self.temporal_attention = torch.nn.Linear(
            64, self.last_layer(self.features(dummy_data)).shape[-1]
        )
        self.feature_attention = torch.nn.Linear(64, configuration.feature_dim)
        log.info(f"Initialized Discriminator with {current_size} features.")

    @staticmethod
    def conv_layer(kernel_size, in_channels, out_channels, pad_type="replicate"):
        def zero_pad_1d(sizes):
            return torch.nn.ConstantPad1d(sizes, 0)

        if pad_type == "reflect":
            pad = torch.nn.ReflectionPad1d
        elif pad_type == "replicate":
            pad = torch.nn.ReplicationPad1d
        elif pad_type == "zero":
            pad = zero_pad_1d

        pad_l = (kernel_size - 1) // 2
        pad_r = kernel_size - 1 - pad_l
        return torch.nn.Sequential(
            pad((pad_l, pad_r)),
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size),
        )

    def forward(
        self,
        input: torch.Tensor,  # concatenation of position and velocity without root
        attention: torch.Tensor,
    ) -> torch.Tensor:

        features = self.last_layer(self.features(input.permute(0, 2, 1)))
        attention_features = self.attention_features(attention.mean(dim=1))
        temporal_attention = self.temporal_attention(attention_features)
        feature_attention = self.feature_attention(attention_features)
        combined_features = (features * feature_attention.unsqueeze(-1)).sum(dim=1)
        final_score = (temporal_attention * combined_features).sum(dim=-1)
        return final_score
