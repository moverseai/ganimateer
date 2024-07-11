import typing

import torch
from moai.nn.convolution import make_conv_block
from moai.utils.iterators import pairwise


class SkeletonBlock(torch.nn.Module):
    def __init__(
        self,
        neighbors: typing.Sequence[typing.Sequence[int]],
        channels: typing.Sequence[int],
        kernel_size: int,
        padding: int,
        padding_mode="zeros",
        bias=True,
        activation_type="lrelu",
    ):
        super().__init__()

        padding = (kernel_size - 1) // 2
        blocks = []
        for cin, cout in pairwise(channels[:-1]):

            block = make_conv_block(
                "conv1d",
                "skeleton",  ### ADD
                cin,
                cout,
                activation_type=activation_type,
                convolution_params={
                    "kernel_size": kernel_size,
                    "neighbors": neighbors,
                    "bias": bias,
                    "padding": padding,
                    "padding_mode": padding_mode,
                },
                activation_params={"inplace": True, "negative_slope": 0.2},
            )

            blocks.append(block)

        pred = make_conv_block(
            "conv1d",
            "conv1d" if channels[-1] == 1 else "skeleton",
            channels[-2],
            channels[-1],
            activation_type="none",
            convolution_params={
                "kernel_size": kernel_size,
                "neighbors": neighbors,
                "bias": bias,
                "padding": padding,
                "padding_mode": padding_mode,
            },
        )
        blocks.append(pred)
        self.layers = torch.nn.Sequential(*blocks)

    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        self.output = input
        return input


def get_channels_list(channels):
    n_channels = channels
    joint_num = n_channels // 6

    base_channel = 128
    n_layers = 4

    channels_list = [n_channels]
    for i in range(n_layers - 1):
        channels_list.append(base_channel * (2 ** ((i + 1) // 2)))
    channels_list += [n_channels]
    channels_list = [((n - 1) // joint_num + 1) * joint_num for n in channels_list]

    return channels_list


def dfs(parents, x, vis, dist):
    fa = parents
    vis[x] = 1
    for y in range(len(fa)):
        if (fa[y] == x or fa[x] == y) and vis[y] == 0:
            dist[y] = dist[x] + 1
            dfs(parents, y, vis, dist)


def get_neighbor(parents, contact_ids, threshold=2.0, enforce_contact=True):
    fa = parents
    neighbor_list = []
    for x in range(0, len(fa)):
        vis = [0 for _ in range(len(fa))]
        dist = [0 for _ in range(len(fa))]
        dfs(parents, x, vis, dist)
        neighbor = []
        for j in range(0, len(fa)):
            if dist[j] <= threshold:
                neighbor.append(j)
        neighbor_list.append(neighbor)

    contact_list = []
    if True:
        for i, p_id in enumerate(contact_ids):
            v_id = len(neighbor_list)
            neighbor_list[p_id].append(v_id)
            neighbor_list.append(neighbor_list[p_id])
            contact_list.append(v_id)

    root_neighbor = neighbor_list[0]
    id_root = len(neighbor_list)

    if enforce_contact:
        root_neighbor = root_neighbor + contact_list
        for j in contact_list:
            neighbor_list[j] = list(set(neighbor_list[j]))

    root_neighbor = list(set(root_neighbor))
    for j in root_neighbor:
        neighbor_list[j].append(id_root)
    root_neighbor.append(id_root)
    neighbor_list.append(root_neighbor)  # Neighbor for root position
    return neighbor_list


class Generator(torch.nn.Module):
    def __init__(
        self,
        parents: typing.Sequence[int],
        contacts: typing.Sequence[int],
        kernel_size: int,
        padding_mode="reflect",
        bias=True,
        stages: int = 2,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.stages = torch.nn.ModuleList()
        neighbors = get_neighbor(parents, contacts, threshold=2, enforce_contact=True)
        num_features = (len(parents) + len(contacts) + 1) * 6
        channels = get_channels_list(num_features)
        for _ in range(stages):
            self.stages.append(
                SkeletonBlock(
                    neighbors,
                    channels,
                    kernel_size=kernel_size,
                    padding_mode=padding_mode,
                    padding=padding,
                    bias=bias,
                    activation_type="lrelu",
                )
            )

    def forward(
        self,
        noise0: torch.Tensor,
        generated: torch.Tensor,
        noise1: torch.Tensor = None,
    ) -> typing.Dict[str, torch.Tensor]:
        out = []
        for i, stage in enumerate(self.stages):
            if i > 0:
                generated = torch.nn.functional.interpolate(
                    generated, size=noise1.shape[-1], mode="linear", align_corners=False
                )
                generated = generated.detach()
                generated = stage(generated + noise1) + generated
            else:
                generated = stage(generated + noise0) + generated
            out.append(generated)
        return {"stage0": out[0], "stage1": out[1] if noise1 is not None else None}


class Discriminator(torch.nn.Module):
    def __init__(
        self,
        parents: typing.Sequence[int],
        contacts: typing.Sequence[int],
        kernel_size: int,
        padding_mode="reflect",
        bias=True,
        stages: int = 2,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.stages = torch.nn.ModuleList()
        neighbors = get_neighbor(parents, contacts, threshold=2, enforce_contact=True)
        num_features = (len(parents) + len(contacts) + 1) * 6
        channels = get_channels_list(num_features)
        channels[-1] = 1
        for _ in range(stages):
            self.stages.append(
                SkeletonBlock(
                    neighbors,
                    channels,
                    kernel_size=kernel_size,
                    padding_mode=padding_mode,
                    padding=padding,
                    bias=bias,
                    activation_type="lrelu",
                )
            )

    def forward(
        self, fake0: torch.Tensor, fake1: torch.Tensor = None
    ) -> typing.Dict[str, torch.Tensor]:
        out = []
        for i, stage in enumerate(self.stages):
            pred = stage(fake0) if i == 0 else stage(fake1)
            out.append(pred)
        return {"stage0": out[0], "stage1": out[1] if fake1 is not None else None}
