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
        batch_size = init_pose.shape[0]
        body_pose = init_pose[:, 3:].clone()
        global_orient = init_pose[:, :3].clone()
        betas = init_betas.clone()

        pred_cam_t = init_cam_t.clone()
        keypoints_2d = keypoints_2d.clone()
        camera_center = camera_center.clone()

        for i in range(self.num_iters):
            body_pose.requires_grad = True
            global_orient.requires_grad = True
            betas.requires_grad = True
            pred_cam_t.requires_grad = True

            smpl_output = self.smpl(betas=betas,
                                     body_pose=body_pose,
                                     global_orient=global_orient,
                                     transl=pred_cam_t)

            model_joints = smpl_output.joints

            loss, loss_dict = temporal_body_fitting_loss(
                body_pose, global_orient, betas, pred_cam_t,
                model_joints, keypoints_2d,
                self.pose_prior,
                focal_length=self.focal_length,
            )

            loss.backward()

            with torch.no_grad():
                body_pose -= self.step_size * body_pose.grad
                global_orient -= self.step_size * global_orient.grad
                betas -= self.step_size * betas.grad
                pred_cam_t -= self.step_size * pred_cam_t.grad

            body_pose.grad = None
            global_orient.grad = None
            betas.grad = None
            pred_cam_t.grad = None

        return body_pose, global_orient, betas, pred_cam_t
