import torch


# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
class RotMatToEuler(torch.nn.Module):
    def __init__(self, convention: str, degrees: bool = True) -> None:
        super().__init__()
        """
        Convert rotations given as rotation matrices to Euler angles in radians/degrees.

        Args:
            convention: Convention string of three uppercase letters.
            degrees: To return the euler angles in degrees or radians
        """
        self.convention = convention
        self.degrees = degrees
        if len(convention) != 3:
            raise ValueError("Convention must have 3 letters.")
        if convention[1] in (convention[0], convention[2]):
            raise ValueError(f"Invalid convention {convention}.")
        for letter in convention:
            if letter not in ("X", "Y", "Z"):
                raise ValueError(f"Invalid letter {letter} in convention string.")

    def _angle_from_tan(
        self, axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
    ) -> torch.Tensor:
        i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
        if horizontal:
            i2, i1 = i1, i2
        even = (axis + other_axis) in ["XY", "YZ", "ZX"]
        if horizontal == even:
            return torch.atan2(data[..., i1], data[..., i2])
        if tait_bryan:
            return torch.atan2(-data[..., i2], data[..., i1])
        return torch.atan2(data[..., i2], -data[..., i1])

    def _index_from_letter(self, letter: str) -> int:
        if letter == "X":
            return 0
        if letter == "Y":
            return 1
        if letter == "Z":
            return 2
        raise ValueError("letter must be either X, Y or Z.")

    def forward(
        self,
        rotmat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            Rotation matrix in a flattened format (..., 9).
        Returns:
            Euler angles as tensor of shape (..., 3).
        """
        if len(rotmat.shape) == 3:
            b, N, _ = rotmat.shape
            matrix = rotmat.reshape(b, N, 3, 3)
        else:
            matrix = rotmat.reshape(-1, 3, 3)

        i0 = self._index_from_letter(self.convention[0])
        i2 = self._index_from_letter(self.convention[2])
        tait_bryan = i0 != i2
        if tait_bryan:
            central_angle = torch.asin(
                matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
            )
        else:
            central_angle = torch.acos(matrix[..., i0, i0])

        o = (
            self._angle_from_tan(
                self.convention[0],
                self.convention[1],
                matrix[..., i2],
                False,
                tait_bryan,
            ),
            central_angle,
            self._angle_from_tan(
                self.convention[2],
                self.convention[1],
                matrix[..., i0, :],
                True,
                tait_bryan,
            ),
        )
        return (
            -1 * torch.rad2deg(torch.stack(o, -1))
            if self.degrees
            else torch.stack(o, -1)
        )
