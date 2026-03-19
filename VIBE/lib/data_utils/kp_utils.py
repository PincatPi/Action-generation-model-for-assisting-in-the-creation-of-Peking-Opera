import numpy as np

def keypoint_hflip(kp, img_width):
    if len(kp.shape) == 2:
        kp[:,0] = (img_width - 1.) - kp[:,0]
    elif len(kp.shape) == 3:
        kp[:, :, 0] = (img_width - 1.) - kp[:, :, 0]
    return kp

def convert_kps(joints2d, src, dst):
    src_names = eval(f'get_{src}_joint_names')()
    dst_names = eval(f'get_{dst}_joint_names')()

    out_joints2d = np.zeros((joints2d.shape[0], len(dst_names), 3))

    for idx, jn in enumerate(dst_names):
        if jn in src_names:
            out_joints2d[:, idx] = joints2d[:, src_names.index(jn)]

    return out_joints2d

def get_perm_idxs(src, dst):
    src_names = eval(f'get_{src}_joint_names')()
    dst_names = eval(f'get_{dst}_joint_names')()
    idxs = [src_names.index(h) for h in dst_names if h in src_names]
    return idxs

def get_spin_joint_names():
    return [
        'right_ankle',
        'right_knee',
        'right_hip',
        'left_hip',
        'left_knee',
        'left_ankle',
        'right_wrist',
        'right_elbow',
        'right_shoulder',
        'left_shoulder',
        'left_elbow',
        'left_wrist',
        'neck',
        'head_top',
        'pelvis',
        'thorax',
        'spine',
        'jaw',
        'head',
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear'
    ]

def get_insta_joint_names():
    return [
        'right_ankle',
        'right_knee',
        'right_hip',
        'left_hip',
        'left_knee',
        'left_ankle',
        'right_wrist',
        'right_elbow',
        'right_shoulder',
        'left_shoulder',
        'left_elbow',
        'left_wrist',
        'neck',
        'head_top',
    ]
