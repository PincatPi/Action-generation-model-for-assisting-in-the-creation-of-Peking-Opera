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
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ], dim=1).view(batch_size, 3, 3)
    return rotMat


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
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}".format(quaternion.shape))
    q = torch.nn.functional.normalize(quaternion, p=2, dim=-1)
    half_angle = torch.atan2(torch.norm(q[..., 1:], p=2, dim=-1), q[..., 0])
    angle = 2.0 * half_angle
    sin_half_angle_over_angle = torch.where(angle > 1e-6, torch.sin(half_angle) / half_angle, 1.0)
    return q[..., 1:] / sin_half_angle_over_angle.unsqueeze(-1) * angle.unsqueeze(-1)


def rotation_matrix_to_quaternion(rotation_matrix: torch.Tensor) -> torch.Tensor:
    if rotation_matrix.shape[-1] == 3:
        rotation_matrix = torch.cat([rotation_matrix, torch.zeros((*rotation_matrix.shape[:-2], 3, 1), device=rotation_matrix.device)], dim=-1)
        rotation_matrix[..., 3, 3] = 1.0
    
    batch_size = rotation_matrix.shape[0]
    
    matrix_diag = rotation_matrix.diagonal(dim1=-2, dim2=-1)
    trace = torch.sum(matrix_diag, dim=-1)
    
    quaternion = torch.zeros((batch_size, 4), dtype=rotation_matrix.dtype, device=rotation_matrix.device)
    
    trace_pos = trace > 0
    trace_not_pos = ~trace_pos
    
    if trace_pos.any():
        s = torch.sqrt(trace[trace_pos] + 1.0) * 2
        quaternion[trace_pos, 0] = 0.25 * s
        quaternion[trace_pos, 1] = (rotation_matrix[trace_pos, 2, 1] - rotation_matrix[trace_pos, 1, 2]) / s
        quaternion[trace_pos, 2] = (rotation_matrix[trace_pos, 0, 2] - rotation_matrix[trace_pos, 2, 0]) / s
        quaternion[trace_pos, 3] = (rotation_matrix[trace_pos, 1, 0] - rotation_matrix[trace_pos, 0, 1]) / s
    
    if trace_not_pos.any():
        diag = matrix_diag[trace_not_pos]
        max_diag = torch.argmax(diag, dim=-1)
        
        i_mask = max_diag == 0
        j_mask = max_diag == 1
        k_mask = max_diag == 2
        
        if i_mask.any():
            s = torch.sqrt(1.0 + diag[i_mask, 0] - diag[i_mask, 1] - diag[i_mask, 2]) * 2
            quaternion[trace_not_pos][i_mask, 0] = (rotation_matrix[trace_not_pos][i_mask, 2, 1] - rotation_matrix[trace_not_pos][i_mask, 1, 2]) / s
            quaternion[trace_not_pos][i_mask, 1] = 0.25 * s
            quaternion[trace_not_pos][i_mask, 2] = (rotation_matrix[trace_not_pos][i_mask, 0, 1] + rotation_matrix[trace_not_pos][i_mask, 1, 0]) / s
            quaternion[trace_not_pos][i_mask, 3] = (rotation_matrix[trace_not_pos][i_mask, 0, 2] + rotation_matrix[trace_not_pos][i_mask, 2, 0]) / s
        
        if j_mask.any():
            s = torch.sqrt(1.0 + diag[j_mask, 1] - diag[j_mask, 0] - diag[j_mask, 2]) * 2
            quaternion[trace_not_pos][j_mask, 0] = (rotation_matrix[trace_not_pos][j_mask, 0, 2] - rotation_matrix[trace_not_pos][j_mask, 2, 0]) / s
            quaternion[trace_not_pos][j_mask, 1] = (rotation_matrix[trace_not_pos][j_mask, 0, 1] + rotation_matrix[trace_not_pos][j_mask, 1, 0]) / s
            quaternion[trace_not_pos][j_mask, 2] = 0.25 * s
            quaternion[trace_not_pos][j_mask, 3] = (rotation_matrix[trace_not_pos][j_mask, 1, 2] + rotation_matrix[trace_not_pos][j_mask, 2, 1]) / s
        
        if k_mask.any():
            s = torch.sqrt(1.0 + diag[k_mask, 2] - diag[k_mask, 0] - diag[k_mask, 1]) * 2
            quaternion[trace_not_pos][k_mask, 0] = (rotation_matrix[trace_not_pos][k_mask, 1, 0] - rotation_matrix[trace_not_pos][k_mask, 0, 1]) / s
            quaternion[trace_not_pos][k_mask, 1] = (rotation_matrix[trace_not_pos][k_mask, 0, 2] + rotation_matrix[trace_not_pos][k_mask, 2, 0]) / s
            quaternion[trace_not_pos][k_mask, 2] = (rotation_matrix[trace_not_pos][k_mask, 1, 2] + rotation_matrix[trace_not_pos][k_mask, 2, 1]) / s
            quaternion[trace_not_pos][k_mask, 3] = 0.25 * s
    
    return quaternion
