import torch
from lib.models.spin import perspective_projection
from lib.models.smpl import JOINT_IDS


def gmof(x, sigma):
    x_squared = x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def angle_prior(pose):
    return torch.exp(
        pose[:, [55 - 3, 58 - 3, 12 - 3, 15 - 3]] * torch.tensor([1., -1., -1, -1.], device=pose.device)) ** 2


def body_fitting_loss(body_pose, betas, model_joints, camera_t, camera_center,
                      joints_2d, joints_conf, pose_prior,
                      focal_length=5000, sigma=100, pose_prior_weight=4.78,
                      shape_prior_weight=5, angle_prior_weight=15.2,
                      output='sum'):
    batch_size = body_pose.shape[0]
    rotation = torch.eye(3, device=body_pose.device).unsqueeze(0).expand(batch_size, -1, -1)
    projected_joints = perspective_projection(model_joints, rotation, camera_t,
                                              focal_length, camera_center)

    reprojection_error = gmof(projected_joints - joints_2d, sigma)
    reprojection_loss = (joints_conf ** 2) * reprojection_error.sum(dim=-1)

    pose_prior_loss = (pose_prior_weight ** 2) * pose_prior(body_pose, betas)

    angle_prior_loss = (angle_prior_weight ** 2) * angle_prior(body_pose).sum(dim=-1)

    shape_prior_loss = (shape_prior_weight ** 2) * (betas ** 2).sum(dim=-1)

    total_loss = reprojection_loss.sum(dim=-1) + pose_prior_loss + angle_prior_loss + shape_prior_loss

    return total_loss


def temporal_body_fitting_loss(body_pose, global_orient, betas, camera_t,
                                model_joints, joints_2d, pose_prior,
                                focal_length=5000, sigma=100, pose_prior_weight=4.78,
                                shape_prior_weight=5, angle_prior_weight=15.2,
                                output='sum'):
    batch_size = body_pose.shape[0]
    rotation = torch.eye(3, device=body_pose.device).unsqueeze(0).expand(batch_size, -1, -1)
    projected_joints = perspective_projection(model_joints, rotation, camera_t,
                                              focal_length)

    reprojection_error = gmof(projected_joints - joints_2d, sigma)
    reprojection_loss = reprojection_error.sum(dim=-1)

    pose_prior_loss = (pose_prior_weight ** 2) * pose_prior(body_pose, betas)

    angle_prior_loss = (angle_prior_weight ** 2) * angle_prior(body_pose).sum(dim=-1)

    shape_prior_loss = (shape_prior_weight ** 2) * (betas ** 2).sum(dim=-1)

    total_loss = reprojection_loss.sum(dim=-1) + pose_prior_loss + angle_prior_loss + shape_prior_loss

    loss_dict = {
        'reprojection_loss': reprojection_loss.mean(),
        'pose_prior_loss': pose_prior_loss.mean(),
        'angle_prior_loss': angle_prior_loss.mean(),
        'shape_prior_loss': shape_prior_loss.mean(),
    }

    return total_loss.sum(), loss_dict
