import torch
import numpy as np


def compute_accel(joints):
    velocities = joints[1:] - joints[:-1]
    acceleration = velocities[1:] - velocities[:-1]
    acceleration_normed = np.linalg.norm(acceleration, axis=2)
    return np.mean(acceleration_normed, axis=1)


def compute_error_accel(joints_gt, joints_pred, vis=None):
    accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

    normed = np.linalg.norm(accel_pred - accel_gt, axis=2)

    if vis is None:
        new_vis = np.ones(len(normed), dtype=bool)
    else:
        invis = np.logical_not(vis)
        invis1 = np.roll(invis, -1)
        invis2 = np.roll(invis, -2)
        new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
        new_vis = np.logical_not(new_invis)

    return np.mean(normed[new_vis], axis=1)


def compute_error_verts(pred_verts, target_verts=None, target_theta=None):
    if target_verts is None and target_theta is None:
        raise ValueError('Either target_verts or target_theta must be provided')

    if target_verts is None:
        from lib.models.smpl import SMPL
        smpl = SMPL()
        target_verts = smpl(torch.tensor(target_theta))

    error = np.mean(np.linalg.norm(pred_verts - target_verts, axis=2), axis=1)
    return error


def batch_compute_similarity_transform_torch(S1, S2):
    batch_size = S1.shape[0]
    S1 = S1.permute(0, 2, 1)
    S2 = S2.permute(0, 2, 1)

    mu1 = S1.mean(dim=2, keepdim=True)
    mu2 = S2.mean(dim=2, keepdim=True)

    S1 = S1 - mu1
    S2 = S2 - mu2

    H = torch.bmm(S1, S2.permute(0, 2, 1))
    U, s, V = torch.svd(H)

    R = torch.bmm(V, U.permute(0, 2, 1))

    var1 = torch.sum(S1 ** 2, dim=(1, 2))
    scale = torch.sum(s, dim=1) / var1

    t = mu2 - scale.view(-1, 1, 1) * torch.bmm(R, mu1)

    return R, scale, t
