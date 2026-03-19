import os
import cv2
import time
import json
import torch
import subprocess
import numpy as np
import os.path as osp
from pytube import YouTube
from collections import OrderedDict

from lib.utils.smooth_bbox import get_smooth_bbox_params, get_all_bbox_params
from lib.data_utils.img_utils import get_single_image_crop_demo
from lib.utils.geometry import rotation_matrix_to_angle_axis
from lib.smplify.temporal_smplify import TemporalSMPLify


def preprocess_video(video, joints2d, bboxes, frames, scale=1.0, crop_size=224):
    if joints2d is not None:
        bboxes, time_pt1, time_pt2 = get_all_bbox_params(joints2d, vis_thresh=0.3)
        bboxes[:,2:] = 150. / bboxes[:,2:]
        bboxes = np.stack([bboxes[:,0], bboxes[:,1], bboxes[:,2], bboxes[:,2]]).T

        video = video[time_pt1:time_pt2]
        joints2d = joints2d[time_pt1:time_pt2]
        frames = frames[time_pt1:time_pt2]

    shape = video.shape

    temp_video = np.zeros((shape[0], crop_size, crop_size, shape[-1]))
    norm_video = torch.zeros(shape[0], shape[-1], crop_size, crop_size)

    for idx in range(video.shape[0]):
        img = video[idx]
        bbox = bboxes[idx]

        j2d = joints2d[idx] if joints2d is not None else None

        norm_img, raw_img, kp_2d = get_single_image_crop_demo(
            img,
            bbox,
            kp_2d=j2d,
            scale=scale,
            crop_size=crop_size)

        if joints2d is not None:
            joints2d[idx] = kp_2d

        temp_video[idx] = raw_img
        norm_video[idx] = norm_img

    temp_video = temp_video.astype(np.uint8)

    return temp_video, norm_video, bboxes, joints2d, frames


def download_youtube_clip(url, download_folder):
    return YouTube(url).streams.first().download(output_path=download_folder)


def smplify_runner(pred_rotmat, pred_betas, pred_cam, j2d, device, batch_size, lr=1.0, opt_steps=1, use_lbfgs=True, pose2aa=True):
    smplify = TemporalSMPLify(
        step_size=lr,
        batch_size=batch_size,
        num_iters=opt_steps,
        focal_length=5000.,
        use_lbfgs=use_lbfgs,
        device=device,
    )
    if pose2aa:
        pred_pose = rotation_matrix_to_angle_axis(pred_rotmat.detach()).reshape(batch_size, -1)
    else:
        pred_pose = pred_rotmat

    pred_cam_t = torch.stack([
        pred_cam[:, 1], pred_cam[:, 2],
        2 * 5000 / (224 * pred_cam[:, 0] + 1e-9)
    ], dim=-1)

    gt_keypoints_2d_orig = j2d
    opt_joint_loss = smplify.get_fitting_loss(
        pred_pose.detach(), pred_betas.detach(),
        pred_cam_t.detach(),
        0.5 * 224 * torch.ones(batch_size, 2, device=device),
        gt_keypoints_2d_orig).mean(dim=-1)

    best_prediction_id = torch.argmin(opt_joint_loss).item()
    pred_betas = pred_betas[best_prediction_id].unsqueeze(0)

    start = time.time()
    output, new_opt_joint_loss = smplify(
        pred_pose.detach(), pred_betas.detach(),
        pred_cam_t.detach(),
        0.5 * 224 * torch.ones(batch_size, 2, device=device),
        gt_keypoints_2d_orig,
    )
    new_opt_joint_loss = new_opt_joint_loss.mean(dim=-1)

    return output


def convert_crop_coords_to_orig_img(bbox, keypoints, crop_size=224):
    for idx in range(keypoints.shape[0]):
        keypoints[idx, :, :2] = keypoints[idx, :, :2] / crop_size * bbox[idx, 2]
        keypoints[idx, :, 0] += bbox[idx, 0] - bbox[idx, 2] / 2
        keypoints[idx, :, 1] += bbox[idx, 1] - bbox[idx, 3] / 2
    return keypoints


def convert_crop_cam_to_orig_img(cam, bbox, crop_size=224, orig_shape=None):
    scale = bbox[..., 2] / crop_size
    cam_cx = bbox[..., 0] - bbox[..., 2] / 2
    cam_cy = bbox[..., 1] - bbox[..., 3] / 2
    orig_cam = np.stack([cam[:, 0] / scale, cam[:, 1] + cam_cx, cam[:, 2] + cam_cy], axis=-1)
    return orig_cam


def prepare_rendering_results(vibe_results, frame_count):
    vertices = []
    cam = []
    for person_id in vibe_results.keys():
        vertices.append(vibe_results[person_id]['verts'])
        cam.append(vibe_results[person_id]['pred_cam'])

    vertices = np.concatenate(vertices, axis=0)
    cam = np.concatenate(cam, axis=0)

    return vertices, cam


def video_to_images(video_file, output_folder=None, return_info=False):
    cap = cv2.VideoCapture(video_file)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    return frames


def images_to_video(images, output_file, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w = images[0].shape[:2]
    out = cv2.VideoWriter(output_file, fourcc, fps, (w, h))
    for img in images:
        out.write(img)
    out.release()


def download_ckpt(use_3dpw=False):
    return 'data/vibe_model_w_3dpw.pth.tar' if use_3dpw else 'data/vibe_model_wo_3dpw.pth.tar'
