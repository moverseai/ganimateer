import torch
from torch.nn.utils.rnn import pack_padded_sequence


class MoveModel(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 259,
        hidden_size: int = 512,
        output_size: int = 512,
    ) -> None:
        super(MoveModel, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Conv1d(input_size, hidden_size, 4, 2, 1),
            torch.nn.Dropout(0.2, inplace=True),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv1d(hidden_size, output_size, 4, 2, 1),
            torch.nn.Dropout(0.2, inplace=True),
            torch.nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = torch.nn.Linear(output_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            res = self.out_net(self.main(x.permute(0, 2, 1)).permute(0, 2, 1))
        return res


class MovementConvEncoder(torch.nn.Module):
    def __init__(
        self,
        ckpt: str,
        input_size: int = 259,
        hidden_size: int = 512,
        output_size: int = 512,
    ):
        super(MovementConvEncoder, self).__init__()
        snapshot = torch.load(ckpt, map_location="cpu")
        self.movement_enc = MoveModel(input_size, hidden_size, output_size)
        self.movement_enc.load_state_dict(snapshot["movement_encoder"])
        self.movement_enc.eval()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.movement_enc(inputs.detach())


class MotionModel(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 259,
        hidden_size: int = 512,
        output_size: int = 512,
    ) -> None:
        super(MotionModel, self).__init__()
        self.input_emb = torch.nn.Linear(input_size, hidden_size)
        self.gru = torch.nn.GRU(
            hidden_size, hidden_size, batch_first=True, bidirectional=True
        )
        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 2, hidden_size),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(hidden_size, output_size),
        )
        self.hidden_size = hidden_size
        self.hidden = torch.nn.Parameter(
            torch.randn((2, 1, self.hidden_size), requires_grad=False)
        )

    def forward(self, x, m_lens: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            num_samples = x.shape[0]
            input_embs = self.input_emb(x)
            hidden = self.hidden.repeat(1, num_samples, 1)

            cap_lens = m_lens.data.tolist()
            emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)

            _, gru_last = self.gru(emb, hidden)
            gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)
            res = self.output_net(gru_last)

        return res


class MotionEncoderBiGRUCo(torch.nn.Module):
    def __init__(
        self,
        ckpt: str,
        input_size: int = 512,
        hidden_size: int = 1024,
        output_size: int = 512,
    ) -> None:
        super(MotionEncoderBiGRUCo, self).__init__()
        snapshot = torch.load(ckpt, map_location="cpu")
        self.motion_enc = MotionModel(input_size, hidden_size, output_size)
        self.motion_enc.load_state_dict(snapshot["motion_encoder"])
        self.motion_enc.eval()

    def forward(self, inputs):
        m_lens = torch.tensor([5] * inputs.shape[0]).to(inputs.device)
        return self.motion_enc(inputs.detach(), m_lens)
