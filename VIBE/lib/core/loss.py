import torch
import torch.nn as nn

from lib.utils.geometry import batch_rodrigues

class VIBELoss(nn.Module):
    def __init__(
            self,
            e_loss_weight=60.,
            e_3d_loss_weight=30.,
            e_pose_loss_weight=1.,
            e_shape_loss_weight=0.001,
            d_motion_loss_weight=1.,
            device='cuda',
    ):
        super(VIBELoss, self).__init__()
        self.e_loss_weight = e_loss_weight
        self.e_3d_loss_weight = e_3d_loss_weight
        self.e_pose_loss_weight = e_pose_loss_weight
        self.e_shape_loss_weight = e_shape_loss_weight
        self.d_motion_loss_weight = d_motion_loss_weight

        self.device = device
        self.criterion_shape = nn.L1Loss().to(self.device)
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        self.criterion_regr = nn.MSELoss().to(self.device)

        self.enc_loss = batch_encoder_disc_l2_loss
        self.dec_loss = batch_adv_disc_l2_loss

    def forward(
            self,
            generator_outputs,
            data_2d,
            data_3d,
            data_body_mosh=None,
            data_motion_mosh=None,
            body_discriminator=None,
            motion_discriminator=None,
    ):
        reduce = lambda x: x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])
        flatten = lambda x: x.reshape(-1)
        accumulate_thetas = lambda x: torch.cat([output['theta'] for output in x],0)

        if data_2d:
            sample_2d_count = data_2d['kp_2d'].shape[0]
            real_2d = torch.cat((data_2d['kp_2d'], data_3d['kp_2d']), 0)
        else:
            sample_2d_count = 0
            real_2d = data_3d['kp_2d']

        real_2d = reduce(real_2d)

        real_3d = reduce(data_3d['kp_3d'])
        data_3d_theta = reduce(data_3d['theta'])

        w_3d = data_3d['w_3d'].type(torch.bool)
        w_smpl = data_3d['w_smpl'].type(torch.bool)

        total_predict_thetas = accumulate_thetas(generator_outputs)

        preds = generator_outputs[-1]

        pred_j3d = preds['kp_3d'][sample_2d_count:]
        pred_theta = preds['theta'][sample_2d_count:]

        theta_size = pred_theta.shape[:2]

        pred_theta = reduce(pred_theta)
        pred_j2d = reduce(preds['kp_2d'])
        pred_j3d = reduce(pred_j3d)

        w_3d = flatten(w_3d)
        w_smpl = flatten(w_smpl)

        pred_theta = pred_theta[w_smpl]
        pred_j3d = pred_j3d[w_3d]
        data_3d_theta = data_3d_theta[w_smpl]
        real_3d = real_3d[w_3d]

        loss_kp_2d = self.criterion_keypoints(pred_j2d, real_2d).sum(dim=-1).mean()
        loss_kp_3d = self.criterion_keypoints(pred_j3d, real_3d).sum(dim=-1).mean()

        loss_shape = self.criterion_shape(pred_theta[:, 76:], data_3d_theta[:, 76:])

        pred_pose = pred_theta[:, :76].reshape(-1, 24, 3)
        real_pose = data_3d_theta[:, :76].reshape(-1, 24, 3)
        pred_rotmat = batch_rodrigues(pred_pose.view(-1, 3)).view(-1, 24, 3, 3)
        real_rotmat = batch_rodrigues(real_pose.view(-1, 3)).view(-1, 24, 3, 3)
        loss_pose = self.criterion_regr(pred_rotmat, real_rotmat)

        loss = self.e_loss_weight * loss_kp_2d + \
               self.e_3d_loss_weight * loss_kp_3d + \
               self.e_pose_loss_weight * loss_pose + \
               self.e_shape_loss_weight * loss_shape

        return loss


def batch_encoder_disc_l2_loss(disc_input):
    return torch.mean(disc_input ** 2)


def batch_adv_disc_l2_loss(disc_input):
    return torch.mean((disc_input - 1) ** 2)
