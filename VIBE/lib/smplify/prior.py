import os
import pickle
import numpy as np
import torch
import torch.nn as nn

DEFAULT_DTYPE = torch.float32


def create_prior(prior_type, **kwargs):
    if prior_type == 'gmm':
        prior = MaxMixturePrior(**kwargs)
    elif prior_type == 'l2':
        return L2Prior(**kwargs)
    elif prior_type == 'angle':
        return SMPLifyAnglePrior(**kwargs)
    elif prior_type == 'none' or prior_type is None:
        def no_prior(*args, **kwargs):
            return 0.0
        prior = no_prior
    else:
        raise ValueError('Prior {}'.format(prior_type) + ' is not implemented')
    return prior


class SMPLifyAnglePrior(nn.Module):
    def __init__(self, dtype=torch.float32, **kwargs):
        super(SMPLifyAnglePrior, self).__init__()

        angle_prior_idxs = np.array([55, 58, 12, 15], dtype=np.int64)
        angle_prior_idxs = torch.tensor(angle_prior_idxs, dtype=torch.long)
        self.register_buffer('angle_prior_idxs', angle_prior_idxs)

    def forward(self, pose, with_global=False):
        if with_global:
            return torch.exp(pose[:, self.angle_prior_idxs]) ** 2
        else:
            return torch.exp(pose[:, self.angle_prior_idxs - 3]) ** 2


class MaxMixturePrior(nn.Module):
    def __init__(self, prior_folder='prior', num_gaussians=8, dtype=DEFAULT_DTYPE, epsilon=1e-16,
                 use_merged=True, **kwargs):
        super(MaxMixturePrior, self).__init__()

        self.dtype = dtype
        self.num_gaussians = num_gaussians
        self.epsilon = epsilon
        self.use_merged = use_merged

        gmm_fn = 'gmm_{:02d}.pkl'.format(num_gaussians)

        with open(os.path.join(prior_folder, gmm_fn), 'rb') as f:
            gmm = pickle.load(f)

        if not isinstance(gmm, dict):
            raise ValueError('Unknown GMM format!')

        self.register_buffer('gmm_mean', torch.tensor(gmm['mean'], dtype=dtype))
        self.register_buffer('gmm_cov', torch.tensor(gmm['cov'], dtype=dtype))
        self.register_buffer('gmm_weights', torch.tensor(gmm['weights'], dtype=dtype))

    def forward(self, pose, betas):
        pose_mean = self.gmm_mean
        pose_cov = self.gmm_cov
        gmm_weights = self.gmm_weights

        batch_size = pose.shape[0]
        num_pose_dims = pose.shape[1]

        pose = pose.unsqueeze(1)
        pose_mean = pose_mean.unsqueeze(0)
        pose_cov = pose_cov.unsqueeze(0)

        diff = pose - pose_mean

        precision = torch.inverse(pose_cov)
        exponent = torch.einsum('bpi,bpij,bpj->bp', diff, precision, diff)
        det = torch.det(pose_cov)
        log_det = torch.log(det + self.epsilon)
        log_prob = -0.5 * (exponent + log_det)

        weighted_log_prob = log_prob + torch.log(gmm_weights)
        log_prob_max, _ = torch.max(weighted_log_prob, dim=1)

        loss = -log_prob_max
        return loss
