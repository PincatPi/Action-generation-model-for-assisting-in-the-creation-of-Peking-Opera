import os
import torch

from lib.core.config import VIBE_DATA_DIR
from lib.models.smpl import SMPL, JOINT_IDS, SMPL_MODEL_DIR
from lib.smplify.losses import temporal_camera_fitting_loss, temporal_body_fitting_loss

from .prior import MaxMixturePrior

def arrange_betas(pose, betas):
    batch_size = pose.shape[0]
    num_video = betas.shape[0]

    video_size = batch_size // num_video
    betas_ext = torch.zeros(batch_size, betas.shape[-1], device=betas.device)
    for i in range(num_video):
        betas_ext[i*video_size:(i+1)*video_size] = betas[i]

    return betas_ext

class TemporalSMPLify():
    def __init__(self,
                 step_size=1e-2,
                 batch_size=66,
                 num_iters=100,
                 focal_length=5000,
                 use_lbfgs=True,
                 device=torch.device('cuda'),
                 max_iter=20):

        self.device = device
        self.focal_length = focal_length
        self.step_size = step_size
        self.max_iter = max_iter
        ign_joints = ['OP Neck', 'OP RHip', 'OP LHip', 'Right Hip', 'Left Hip']
        self.ign_joints = [JOINT_IDS[i] for i in ign_joints]
        self.num_iters = num_iters

        self.pose_prior = MaxMixturePrior(prior_folder=VIBE_DATA_DIR,
                                          num_gaussians=8,
                                          dtype=torch.float32).to(device)
        self.use_lbfgs = use_lbfgs
        self.smpl = SMPL(SMPL_MODEL_DIR,
                         batch_size=batch_size,
                         create_transl=False).to(self.device)

    def __call__(self, init_pose, init_betas, init_cam_t, camera_center, keypoints_2d):
        camera_translation = init_cam_t.clone()

        joints_2d = keypoints_2d[:, :, :2]
        joints_conf = keypoints_2d[:, :, -1]

        body_pose = init_pose[:, 3:].detach().clone()
        global_orient = init_pose[:, :3].detach().clone()
        betas = init_betas.detach().clone()

        body_pose.requires_grad = False
        betas.requires_grad = False
        global_orient.requires_grad = True
        camera_translation.requires_grad = True

        camera_opt_params = [global_orient, camera_translation]

        if self.use_lbfgs:
            camera_optimizer = torch.optim.LBFGS(camera_opt_params, max_iter=self.max_iter,
                                                 lr=self.step_size, line_search_fn='strong_wolfe')
            for i in range(self.num_iters):
                def closure():
                    camera_optimizer.zero_grad()
                    smpl_output = self.smpl(global_orient=global_orient,
                                           body_pose=body_pose,
                                           betas=betas)
                    model_joints = smpl_output.joints
                    loss = temporal_camera_fitting_loss(
                        model_joints, camera_translation,
                        init_cam_t, joints_2d, joints_conf,
                        camera_center, self.focal_length,
                        self.ign_joints
                    )
                    loss.backward()
                    return loss
                camera_optimizer.step(closure)

        return {'cam_t': camera_translation, 'global_orient': global_orient}

    def get_fitting_loss(self, pose, betas, cam_t, camera_center, keypoints_2d):
        smpl_output = self.smpl(global_orient=pose[:, :3],
                               body_pose=pose[:, 3:],
                               betas=betas)
        model_joints = smpl_output.joints
        loss = temporal_camera_fitting_loss(
            model_joints, cam_t,
            cam_t, keypoints_2d[:, :, :2], keypoints_2d[:, :, -1],
            camera_center, self.focal_length,
            self.ign_joints
        )
        return loss
