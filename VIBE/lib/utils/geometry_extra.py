def rotation_matrix_to_angle_axis(rotation_matrix):
    if rotation_matrix.shape[1:] == (3,3):
        rot_mat = rotation_matrix.reshape(-1, 3, 3)
        hom = torch.tensor([0, 0, 1], dtype=torch.float32,
                           device=rotation_matrix.device).reshape(1, 3, 1).expand(rot_mat.shape[0], -1, -1)
        rotation_matrix = torch.cat([rot_mat, hom], dim=-1)

    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    aa = quaternion_to_angle_axis(quaternion)
    aa[torch.isnan(aa)] = 0.0
    return aa


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))

    q = torch.nn.functional.normalize(quaternion, p=2, dim=-1)

    half_angle = torch.atan2(torch.norm(q[..., 1:], p=2, dim=-1), q[..., 0])
    angle = 2.0 * half_angle

    sin_half_angle_over_angle = torch.where(
        angle.abs() > 1e-4,
        torch.sin(half_angle) / angle,
        torch.ones_like(angle) * 0.5
    )

    return q[..., 1:] / sin_half_angle_over_angle.unsqueeze(-1)


def rot6d_to_rotmat(x):
    x = x.reshape(-1, 3, 2)

    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = torch.nn.functional.normalize(a1, dim=1)
    b2 = a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack((b1, b2, b3), dim=-1)


def rotmat_to_rot6d(x):
    return x[:, :, :2].reshape(-1, 6)
