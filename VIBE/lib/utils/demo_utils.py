import os
import cv2
import torch
import numpy as np
import os.path as osp
from collections import OrderedDict

from lib.utils.smooth_bbox import get_smooth_bbox_params, get_all_bbox_params
from lib.data_utils.img_utils import get_single_image_crop_demo


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

    for idx in range(shape[0]):
        img = video[idx]
        bbox = bboxes[idx]

        norm_img, _ = get_single_image_crop_demo(
            img,
            bbox,
            kp_2d=joints2d[idx] if joints2d is not None else None,
            scale=scale,
            crop_size=crop_size)

        temp_video[idx] = norm_img
        norm_video[idx] = torch.from_numpy(norm_img.astype(np.float32)).permute(2,0,1) / 255.

    return temp_video, norm_video, bboxes, joints2d, frames


def video_to_images(video_file):
    cap = cv2.VideoCapture(video_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    image_folder = video_file.replace('.mp4', '')
    os.makedirs(image_folder, exist_ok=True)

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(osp.join(image_folder, f'{i:06d}.jpg'), frame)

    cap.release()
    return image_folder, frame_count, (height, width)


def images_to_video(image_folder, output_file, fps=30):
    images = sorted([img for img in os.listdir(image_folder) if img.endswith('.png') or img.endswith('.jpg')])

    frame = cv2.imread(osp.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        video.write(cv2.imread(osp.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
