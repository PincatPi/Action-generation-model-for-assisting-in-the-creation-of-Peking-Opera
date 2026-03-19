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
    if target_verts is None:
        from lib.models.smpl import SMPL_MODEL_DIR
        from lib.models.smpl import SMPL
        device = 'cpu'
        smpl = SMPL(SMPL_MODEL_DIR, batch_size=1).to(device)

        betas = torch.from_numpy(target_theta[:,75:]).to(device)
        pose = torch.from_numpy(target_theta[:,3:75]).to(device)

        target_verts = []
        b_ = torch.split(betas, 5000)
        p_ = torch.split(pose, 5000)

        for b,p in zip(b_,p_):
            output = smpl(betas=b, body_pose=p[:, 3:], global_orient=p[:, :3], pose2rot=True)
            target_verts.append(output.vertices.detach().cpu().numpy())

        target_verts = np.concatenate(target_verts, axis=0)

    error = np.mean(np.linalg.norm(pred_verts - target_verts, axis=-1))
    return error


def batch_compute_similarity_transform_torch(S1, S2):
    batch_size = S1.shape[0]
    S1 = S1.permute(0, 2, 1)
    S2 = S2.permute(0, 2, 1)

    mu1 = S1.mean(dim=2, keepdim=True)
    mu2 = S2.mean(dim=2, keepdim=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    var1 = torch.sum(X1**2, dim=2).sum(dim=1)
    std1 = torch.sqrt(var1 + 1e-6)

    X1 = X1 / std1.unsqueeze(-1).unsqueeze(-1)
    X2 = X2 / std1.unsqueeze(-1).unsqueeze(-1)

    R = torch.matmul(X1, X2.permute(0, 2, 1))
    R = R.cpu().numpy()

    S1 = S1.cpu().numpy()
    S2 = S2.cpu().numpy()

    out = []
    for i in range(batch_size):
        U, s, Vt = np.linalg.svd(R[i])
        r = np.dot(U, Vt)
        s = np.linalg.det(r)
        s = np.diag([1, 1, s])
        r = np.dot(U, np.dot(s, Vt))
        out.append(r)

    R = np.stack(out, axis=0)
    R = torch.from_numpy(R).float().cuda()

    S1 = torch.from_numpy(S1).float().cuda()
    S2 = torch.from_numpy(S2).float().cuda()

    S1 = torch.matmul(R, S1)

    S1 = S1.permute(0, 2, 1)

    return S1
