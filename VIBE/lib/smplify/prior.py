from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import time
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

        angle_prior_signs = np.array([1, -1, -1, -1],
                                     dtype=np.float32 if dtype == torch.float32
                                     else np.float64)
        angle_prior_signs = torch.tensor(angle_prior_signs, dtype=dtype)
        self.register_buffer('angle_prior_signs', angle_prior_signs)

    def forward(self, pose, with_global_pose=False):
        if with_global_pose:
            angle_prior_idxs = self.angle_prior_idxs
        else:
            angle_prior_idxs = self.angle_prior_idxs - 3
        return torch.exp(pose[:, angle_prior_idxs] * self.angle_prior_signs) ** 2


class L2Prior(nn.Module):
    def __init__(self, dtype=torch.float32, **kwargs):
        super(L2Prior, self).__init__()

    def forward(self, module_input):
        return torch.sum(module_input ** 2, dim=-1)


class MaxMixturePrior(nn.Module):
    def __init__(self, prior_folder='prior', num_gaussians=8, dtype=torch.float32, epsilon=1e-16,
                 use_merged=True, **kwargs):
        super(MaxMixturePrior, self).__init__()

        self.num_gaussians = num_gaussians
        self.epsilon = epsilon
        self.use_merged = use_merged
        gmm_fn = 'gmm_{:02d}.pkl'.format(num_gaussians)

        full_gmm_fn = os.path.join(prior_folder, gmm_fn)
        if not os.path.exists(full_gmm_fn):
            print('Could not find {}!'.format(full_gmm_fn))

        with open(full_gmm_fn, 'rb') as f:
            gmm = pickle.load(f, encoding='latin1')

        if not isinstance(gmm, dict):
            gmm = {'means': gmm['means'], 'covariances': gmm['covars']}

        self.register_buffer('gmm_means', torch.tensor(gmm['means'], dtype=dtype))
        self.register_buffer('gmm_covariances', torch.tensor(gmm['covariances'], dtype=dtype))

        self.register_buffer('weights', torch.tensor(gmm['weights'], dtype=dtype))

    def forward(self, pose, betas):
        pose = pose.reshape(-1, 3)
        batch_size = pose.shape[0]
        num_pose = pose.shape[1]
        pose = pose.unsqueeze(1).expand(batch_size, self.num_gaussians, -1)
        gmm_means = self.gmm_means.unsqueeze(0).expand(batch_size, -1, -1)
        gmm_covariances = self.gmm_covariances.unsqueeze(0).expand(batch_size, -1, -1, -1)
        weights = self.weights.unsqueeze(0).expand(batch_size, -1)
        diff = pose - gmm_means
        precision = torch.inverse(gmm_covariances)
        mahalanobis = torch.einsum('bni,bnij,bnj->bn', diff, precision, diff)
        log_prob = -0.5 * mahalanobis + torch.log(weights + self.epsilon)
        log_prob = torch.logsumexp(log_prob, dim=1)
        return -log_prob
