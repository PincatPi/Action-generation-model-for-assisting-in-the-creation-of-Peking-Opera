import numpy as np
import scipy.signal as signal
from scipy.ndimage.filters import gaussian_filter1d


def get_smooth_bbox_params(kps, vis_thresh=2, kernel_size=11, sigma=3):
    bbox_params, start, end = get_all_bbox_params(kps, vis_thresh)
    smoothed = smooth_bbox_params(bbox_params, kernel_size, sigma)
    smoothed = np.vstack((np.zeros((start, 3)), smoothed))
    return smoothed, start, end


def kp_to_bbox_param(kp, vis_thresh):
    if kp is None:
        return
    vis = kp[:, 2] > vis_thresh
    if not np.any(vis):
        return
    vis_kp = kp[vis, :2]
    min_pt = np.min(vis_kp, axis=0)
    max_pt = np.max(vis_kp, axis=0)
    person_height = np.linalg.norm(max_pt - min_pt)
    if person_height < 0.5:
        return
    center = (min_pt + max_pt) / 2.
    scale = 150. / person_height
    return np.concatenate([center, scale.reshape(1)])


def get_all_bbox_params(kps, vis_thresh=2):
    bbox_params = []
    start = 0
    end = len(kps)
    for i, kp in enumerate(kps):
        param = kp_to_bbox_param(kp, vis_thresh)
        if param is None:
            bbox_params.append(None)
        else:
            bbox_params.append(param)

    bbox_params = np.array(bbox_params)
    return bbox_params, start, end


def smooth_bbox_params(bbox_params, kernel_size=11, sigma=3):
    bbox_params = np.array(bbox_params)
    smoothed = np.zeros_like(bbox_params)

    for i in range(3):
        smoothed[:, i] = signal.medfilt(bbox_params[:, i], kernel_size)
        smoothed[:, i] = gaussian_filter1d(smoothed[:, i], sigma=sigma)

    return smoothed
