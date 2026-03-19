import torch
import numpy as np
from torch.nn import functional as F


def batch_rodrigues(axisang):
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    rot_mat = quat2mat(quat)
    rot_mat = rot_mat.view(rot_mat.shape[0], 9)
    return rot_mat


def quat2mat(quat):
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
        2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
        2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2
    ], dim=1).view(batch_size, 3, 3)

    return rotMat


def rotation_matrix_to_angle_axis(rotation_matrix):
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    aa = quaternion_to_angle_axis(quaternion)
    return aa


def rotation_matrix_to_quaternion(rotation_matrix):
    batch_size = rotation_matrix.shape[0]
    num_rotations = rotation_matrix.shape[1] // 3
    rotation_matrix = rotation_matrix.view(batch_size, num_rotations, 3, 3)

    quaternion = torch.zeros(batch_size, num_rotations, 4).to(rotation_matrix.device)

    for i in range(num_rotations):
        rmat = rotation_matrix[:, i, :, :]
        batch_size = rmat.shape[0]
        mask0 = (rmat[:, 0, 0] > rmat[:, 1, 1]) & (rmat[:, 0, 0] > rmat[:, 2, 2])
        mask1 = (~mask0) & (rmat[:, 1, 1] > rmat[:, 2, 2])
        mask2 = (~mask0) & (~mask1)

        quaternion[mask0, i, 0] = torch.sqrt(1 + rmat[mask0, 0, 0] - rmat[mask0, 1, 1] - rmat[mask0, 2, 2]) / 2
        quaternion[mask0, i, 1] = (rmat[mask0, 1, 0] - rmat[mask0, 0, 1]) / (4 * quaternion[mask0, i, 0])
        quaternion[mask0, i, 2] = (rmat[mask0, 0, 2] - rmat[mask0, 2, 0]) / (4 * quaternion[mask0, i, 0])
        quaternion[mask0, i, 3] = (rmat[mask0, 2, 1] - rmat[mask0, 1, 2]) / (4 * quaternion[mask0, i, 0])

        quaternion[mask1, i, 0] = (rmat[mask1, 1, 0] - rmat[mask1, 0, 1]) / (4 * quaternion[mask1, i, 1])
        quaternion[mask1, i, 1] = torch.sqrt(1 - rmat[mask1, 0, 0] + rmat[mask1, 1, 1] - rmat[mask1, 2, 2]) / 2
        quaternion[mask1, i, 2] = (rmat[mask1, 2, 1] - rmat[mask1, 1, 2]) / (4 * quaternion[mask1, i, 1])
        quaternion[mask1, i, 3] = (rmat[mask1, 0, 2] - rmat[mask1, 2, 0]) / (4 * quaternion[mask1, i, 1])

        quaternion[mask2, i, 0] = (rmat[mask2, 0, 2] - rmat[mask2, 2, 0]) / (4 * quaternion[mask2, i, 2])
        quaternion[mask2, i, 1] = (rmat[mask2, 2, 1] - rmat[mask2, 1, 2]) / (4 * quaternion[mask2, i, 2])
        quaternion[mask2, i, 2] = torch.sqrt(1 - rmat[mask2, 0, 0] - rmat[mask2, 1, 1] + rmat[mask2, 2, 2]) / 2
        quaternion[mask2, i, 3] = (rmat[mask2, 1, 0] - rmat[mask2, 0, 1]) / (4 * quaternion[mask2, i, 2])

    return quaternion.view(batch_size, -1)


def quaternion_to_angle_axis(quaternion):
    quaternion = quaternion / torch.norm(quaternion, dim=1, keepdim=True)
    angle = 2 * torch.acos(quaternion[:, 0:1])
    axis = quaternion[:, 1:4] / torch.sin(angle / 2 + 1e-8)
    angle_axis = angle * axis
    return angle_axis
