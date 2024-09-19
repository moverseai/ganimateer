import torch


class LinearVelocityXZ(torch.nn.Module):
    def __init__(self) -> None:
        super(LinearVelocityXZ, self).__init__()

    @torch.no_grad()
    def _qrot(self, q, v):
        """
        Rotate vector(s) v about the rotation described by quaternion(s) q.
        Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
        where * denotes any number of dimensions.
        Returns a tensor of shape (*, 3).
        """
        assert q.shape[-1] == 4
        assert v.shape[-1] == 3
        assert q.shape[:-1] == v.shape[:-1]

        original_shape = list(v.shape)
        # print(q.shape)
        q = q.contiguous().view(-1, 4)
        v = v.contiguous().view(-1, 3)

        qvec = q[:, 1:]
        uv = torch.cross(qvec, v, dim=1)
        uuv = torch.cross(qvec, uv, dim=1)
        return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)

    def forward(
        self,
        velocity: torch.Tensor,  # [B, T, 1, 3]
        rotations: torch.Tensor,  # [B, T, 1, 4]
    ) -> torch.Tensor:
        velocity = self._qrot(rotations, velocity)
        return velocity[..., [0, 2]]


class AngularVelocity(torch.nn.Module):
    def __init__(self, zero_pad: bool = True) -> None:
        super(AngularVelocity, self).__init__()
        self.zero_pad = zero_pad

    @torch.no_grad()
    def _qmul(self, q, r):
        """
        Multiply quaternion(s) q with quaternion(s) r.
        Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
        Returns q*r as a tensor of shape (*, 4).
        """
        assert q.shape[-1] == 4
        assert r.shape[-1] == 4

        original_shape = q.shape

        # Compute outer product
        terms = torch.bmm(r.reshape(-1, 4, 1), q.reshape(-1, 1, 4))

        w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
        x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
        y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
        z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]

        return torch.stack((w, x, y, z), dim=1).view(original_shape)

    @torch.no_grad()
    def _qinv(self, q):
        assert q.shape[-1] == 4, "q must be a tensor of shape (*, 4)"
        mask = torch.ones_like(q)
        mask[..., 1:] = -mask[..., 1:]
        return q * mask

    def forward(
        self, rotations: torch.Tensor
    ) -> torch.Tensor:  # [B, (T), J, 4] quaternion repr expected
        r_velocity = self._qmul(rotations[:, 1:], self._qinv(rotations[:, :-1]))
        if self.zero_pad:
            pad = (
                torch.zeros_like(r_velocity[:1, :])
                if r_velocity.ndim <= 3
                else torch.zeros_like(r_velocity[:, :1, :])
            )
            r_velocity = torch.cat(
                [pad, r_velocity], dim=0 if r_velocity.ndim <= 3 else 1
            )

        return torch.arcsin(r_velocity[..., 2:3])
