# -*- coding: utf-8 -*-

import torch
import numpy as np

from lib.models.smpl import SMPL, SMPL_MODEL_DIR
from lib.utils.one_euro_filter import OneEuroFilter


def smooth_pose(pred_pose, pred_betas, min_cutoff=0.004, beta=0.7):
    one_euro_filter = OneEuroFilter(
        np.zeros_like(pred_pose[0]),
        pred_pose[0],
        min_cutoff=min_cutoff,
        beta=beta,
    )

    smpl = SMPL(model_path=SMPL_MODEL_DIR)

    pred_pose_hat = np.zeros_like(pred_pose)

    pred_pose_hat[0] = pred_pose[0]

    pred_verts_hat = []
    pred_joints3d_hat = []

    smpl_output = smpl(
        betas=torch.from_numpy(pred_betas[0]).unsqueeze(0),
        body_pose=torch.from_numpy(pred_pose[0, 1:]).unsqueeze(0),
        global_orient=torch.from_numpy(pred_pose[0, 0:1]).unsqueeze(0),
    )
    pred_verts_hat.append(smpl_output.vertices.detach().cpu().numpy())
    pred_joints3d_hat.append(smpl_output.joints.detach().cpu().numpy())

    for idx, pose in enumerate(pred_pose[1:]):
        idx += 1

        t = np.ones_like(pose) * idx
        pose = one_euro_filter(t, pose)
        pred_pose_hat[idx] = pose

        smpl_output = smpl(
            betas=torch.from_numpy(pred_betas[idx]).unsqueeze(0),
            body_pose=torch.from_numpy(pred_pose_hat[idx, 1:]).unsqueeze(0),
            global_orient=torch.from_numpy(pred_pose_hat[idx, 0:1]).unsqueeze(0),
        )
        pred_verts_hat.append(smpl_output.vertices.detach().cpu().numpy())
        pred_joints3d_hat.append(smpl_output.joints.detach().cpu().numpy())

    return np.vstack(pred_verts_hat), pred_pose_hat, np.vstack(pred_joints3d_hat)


def smooth_pose_and_cam(pred_pose, pred_betas, pred_cam, min_cutoff=0.004, beta=0.7):
    pred_verts_hat, pred_pose_hat, pred_joints3d_hat = smooth_pose(
        pred_pose, pred_betas, min_cutoff=min_cutoff, beta=beta
    )

    cam_filter = OneEuroFilter(
        np.zeros_like(pred_cam[0]),
        pred_cam[0],
        min_cutoff=min_cutoff,
        beta=beta,
    )
    pred_cam_hat = np.zeros_like(pred_cam)
    pred_cam_hat[0] = pred_cam[0]

    for idx in range(1, len(pred_cam)):
        t = np.ones_like(pred_cam[0]) * idx
        pred_cam_hat[idx] = cam_filter(t, pred_cam[idx])

    return pred_verts_hat, pred_pose_hat, pred_joints3d_hat, pred_cam_hat
