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

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def generate_patch_image_cv(cvimg, c_x, c_y, bb_width, bb_height, patch_width, patch_height, do_flip, scale, rot):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    if do_flip:
        img = img[:, ::-1, :]
        c_x = img_width - c_x - 1

    trans = gen_trans_from_patch_cv(c_x, c_y, bb_width, bb_height, patch_width, patch_height, scale, rot, inv=False)

    img_patch = cv2.warpAffine(img, trans, (int(patch_width), int(patch_height)),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return img_patch, trans

def get_single_image_crop(image, bbox, scale=1.2, crop_size=224):
    if isinstance(image, str):
        image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
    else:
        image = image.copy()

    bbox_x, bbox_y, bbox_w, bbox_h = bbox
    center = [bbox_x + bbox_w/2, bbox_y + bbox_h/2]
    width = bbox_w * scale
    height = bbox_h * scale

    trans = gen_trans_from_patch_cv(center[0], center[1], width, height, crop_size, crop_size, 1.0, 0.0, inv=False)
    img_patch = cv2.warpAffine(image, trans, (crop_size, crop_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    img_patch = img_patch[:, :, ::-1].copy()
    img_patch = transforms.ToTensor()(img_patch)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_patch = normalize(img_patch)

    return img_patch

def get_single_image_crop_demo(image, bbox, kp_2d, scale=1.2, crop_size=224):
    if isinstance(image, str):
        image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
    else:
        image = image.copy()

    bbox_x, bbox_y, bbox_w, bbox_h = bbox
    center = [bbox_x + bbox_w/2, bbox_y + bbox_h/2]
    width = bbox_w * scale
    height = bbox_h * scale

    trans = gen_trans_from_patch_cv(center[0], center[1], width, height, crop_size, crop_size, 1.0, 0.0, inv=False)
    img_patch = cv2.warpAffine(image, trans, (crop_size, crop_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    if kp_2d is not None:
        for i in range(len(kp_2d)):
            kp_2d[i, 0:2] = trans_point2d(kp_2d[i, 0:2], trans)

    img_patch = img_patch[:, :, ::-1].copy()
    img_patch = transforms.ToTensor()(img_patch)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_patch = normalize(img_patch)

    return img_patch, image, kp_2d
