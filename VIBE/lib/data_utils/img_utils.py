import os
import cv2
import torch
import random
import numpy as np
import torchvision.transforms as transforms
from skimage.util.shape import view_as_windows

def get_image(filename):
    image = cv2.imread(filename)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def do_augmentation(scale_factor=0.3, color_factor=0.2):
    scale = random.uniform(1.2, 1.2+scale_factor)
    rot = 0
    do_flip = False
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = [random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)]
    return scale, rot, do_flip, color_scale

def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.zeros(2)
    src_center[0] = c_x
    src_center[1] = c_y
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    trans = cv2.getAffineTransform(src, dst)

    if inv:
        trans = cv2.invertAffineTransform(trans)

    return trans

def get_single_image_crop(image, bbox, scale=1.2, crop_size=224):
    bbox = np.array(bbox).reshape(4)
    center = bbox[:2]
    bbox_size = bbox[2:]
    aspect_ratio = crop_size[1] / crop_size[0]

    trans = gen_trans_from_patch_cv(
        center[0], center[1], bbox_size[0], bbox_size[1],
        crop_size[0], crop_size[1], scale, 0, inv=False)

    cropped_image = cv2.warpAffine(image, trans, (crop_size[0], crop_size[1]), flags=cv2.INTER_LINEAR)

    return cropped_image

def get_single_image_crop_demo(image, bbox, kp_2d=None, scale=1.2, crop_size=224):
    bbox = np.array(bbox).reshape(4)
    center = bbox[:2]
    bbox_size = bbox[2:]

    trans = gen_trans_from_patch_cv(
        center[0], center[1], bbox_size[0], bbox_size[1],
        crop_size, crop_size, scale, 0, inv=False)

    cropped_image = cv2.warpAffine(image, trans, (crop_size, crop_size), flags=cv2.INTER_LINEAR)

    if kp_2d is not None:
        kp_2d[:, :2] = trans_point2d(kp_2d[:, :2].T, trans).T
        return cropped_image, kp_2d

    return cropped_image

def split_into_chunks(vid_names, seqlen, stride):
    vid_names = np.array(vid_names)
    N = len(vid_names)
    vid_ids = np.zeros(N)
    curr_id = 0
    for i in range(N):
        if i > 0 and vid_names[i] != vid_names[i-1]:
            curr_id += 1
        vid_ids[i] = curr_id

    curr_id = 0
    indices = []
    for i in range(N):
        if vid_ids[i] == curr_id:
            indices.append(i)
        else:
            curr_id += 1
            indices.append(i)

    chunks = []
    for i in range(0, N, stride):
        if i + seqlen <= N:
            if vid_ids[i] == vid_ids[i + seqlen - 1]:
                chunks.append([i, i + seqlen - 1])

    return chunks
