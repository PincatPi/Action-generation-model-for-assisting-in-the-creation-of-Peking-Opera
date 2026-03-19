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

    for idx in range(1, pred_pose.shape[0]):
        pred_pose_hat[idx], _ = one_euro_filter(pred_pose[idx], pred_pose[idx])

        smpl_output = smpl(
            betas=torch.from_numpy(pred_betas[idx]).unsqueeze(0),
            body_pose=torch.from_numpy(pred_pose_hat[idx, 1:]).unsqueeze(0),
            global_orient=torch.from_numpy(pred_pose_hat[idx, 0:1]).unsqueeze(0),
        )
        pred_verts_hat.append(smpl_output.vertices.detach().cpu().numpy())
        pred_joints3d_hat.append(smpl_output.joints.detach().cpu().numpy())

    pred_verts_hat = np.concatenate(pred_verts_hat, axis=0)
    pred_joints3d_hat = np.concatenate(pred_joints3d_hat, axis=0)

    return pred_pose_hat, pred_verts_hat, pred_joints3d_hat
