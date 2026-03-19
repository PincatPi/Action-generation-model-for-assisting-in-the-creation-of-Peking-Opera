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
        angle_prior_idxs = self.angle_prior_idxs - (not with_global_pose) * 3
        return torch.exp(pose[:, angle_prior_idxs] *
                         self.angle_prior_signs).pow(2)


class L2Prior(nn.Module):
    def __init__(self, dtype=DEFAULT_DTYPE, reduction='sum', **kwargs):
        super(L2Prior, self).__init__()

    def forward(self, module_input, *args):
        return torch.sum(module_input.pow(2))


class MaxMixturePrior(nn.Module):
    def __init__(self, prior_folder='prior',
                 num_gaussians=6, dtype=DEFAULT_DTYPE, epsilon=1e-16,
                 use_merged=True, **kwargs):
        super(MaxMixturePrior, self).__init__()

        if dtype == DEFAULT_DTYPE:
            np_dtype = np.float32
        elif dtype == torch.float64:
            np_dtype = np.float64
        else:
            print('Unknown float type {}, exiting!'.format(dtype))
            sys.exit(-1)

        self.num_gaussians = num_gaussians
        self.epsilon = epsilon
        self.use_merged = use_merged
        gmm_fn = 'gmm_{:02d}.pkl'.format(num_gaussians)

        full_gmm_fn = os.path.join(prior_folder, gmm_fn)
        if not os.path.exists(full_gmm_fn):
            print('The path to the mixture prior "{}"'.format(full_gmm_fn) +
                  ' does not exist, exiting!')
            sys.exit(-1)

        with open(full_gmm_fn, 'rb') as f:
            gmm = pickle.load(f, encoding='latin1')

        if type(gmm) == dict:
            means = gmm['means'].astype(np_dtype)
            covs = gmm['covars'].astype(np_dtype)
            weights = gmm['weights'].astype(np_dtype)
        elif 'sklearn.mixture.gmm.GMM' in str(type(gmm)):
            means = gmm.means_.astype(np_dtype)
            covs = gmm.covars_.astype(np_dtype)
            weights = gmm.weights_.astype(np_dtype)
        else:
            print('Unknown type for the prior: {}, exiting!'.format(type(gmm)))
            sys.exit(-1)

        self.register_buffer('means', torch.tensor(means, dtype=dtype))
        self.register_buffer('covs', torch.tensor(covs, dtype=dtype))

        precisions = [np.linalg.inv(cov) for cov in covs]
        precisions = np.stack(precisions).astype(np_dtype)
        self.register_buffer('precisions', torch.tensor(precisions, dtype=dtype))

        self.register_buffer('weights', torch.tensor(weights, dtype=dtype))

    def forward(self, pose, betas):
        batch_size = pose.shape[0]
        pose = pose.unsqueeze(1)
        diff = pose - self.means.unsqueeze(0)

        return diff
