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
        smpl = SMPL(
            SMPL_MODEL_DIR,
            batch_size=1,
        ).to(device)

        betas = torch.from_numpy(target_theta[:,75:]).to(device)
        pose = torch.from_numpy(target_theta[:,3:75]).to(device)

        target_verts = []
        b_ = torch.split(betas, 5000)
        p_ = torch.split(pose, 5000)

        for b,p in zip(b_,p_):
            output = smpl(betas=b, body_pose=p[:, 3:], global_orient=p[:, :3], pose2rot=True)
            target_verts.append(output.vertices.detach().cpu().numpy())

        target_verts = np.concatenate(target_verts, axis=0)

    assert len(pred_verts) == len(target_verts)
    error_per_vert = np.sqrt(np.sum((target_verts - pred_verts) ** 2, axis=2))
    return np.mean(error_per_vert, axis=1)


def compute_similarity_transform(S1, S2):
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    var1 = np.sum(X1**2)

    K = X1.dot(X2.T)

    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    R = V.dot(Z.dot(U.T))

    scale = np.trace(R.dot(K)) / var1

    t = mu2 - scale*(R.dot(mu1))

    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def compute_similarity_transform_torch(S1, S2):
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert (S2.shape[1] == S1.shape[1])

    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    var1 = torch.sum(X1 ** 2)

    K = X1.mm(X2.T)

    U, s, V = torch.svd(K)
    Z = torch.eye(U.shape[0], device=S1.device)
    Z[-1, -1] *= torch.sign(torch.det(U @ V.T))
    R = V.mm(Z.mm(U.T))

    scale = torch.trace(R.mm(K)) / var1
    t = mu2 - scale * (R.mm(mu1))

    S1_hat = scale * R.mm(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def batch_compute_similarity_transform_torch(S1, S2):
    transposed = False
    if S1.shape[-1] != 3 and S1.shape[-1] != 2:
        S1 = S1.permute(0, 2, 1)
        S2 = S2.permute(0, 2, 1)
        transposed = True

    mu1 = S1.mean(axis=2, keepdims=True)
    mu2 = S2.mean(axis=2, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    var1 = torch.sum(X1 ** 2, dim=1).sum(dim=1)

    K = torch.bmm(X1, X2.permute(0, 2, 1))

    U, s, V = torch.svd(K)
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0], 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.det(torch.bmm(U, V.permute(0, 2, 1))))
    R = torch.bmm(V, torch.bmm(Z, U.permute(0, 2, 1)))

    var1 = var1.unsqueeze(-1)
    scale = torch.einsum('bii->b', R.bmm(K)) / var1

    t = mu2 - scale.unsqueeze(-1).unsqueeze(-1) * torch.bmm(R, mu1)

    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * torch.bmm(R, S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0, 2, 1)

    return S1_hat
