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

def get_mpii3d_test_joint_names():
    return ['headtop', 'neck', 'rshoulder', 'relbow', 'rwrist', 'lshoulder', 'lelbow', 'lwrist', 'rhip', 'rknee', 'rankle', 'lhip', 'lknee', 'lankle', 'hip', 'Spine (H36M)', 'Head (H36M)']

def get_mpii3d_joint_names():
    return ['spine3', 'spine4', 'spine2', 'Spine (H36M)', 'hip', 'neck', 'Head (H36M)', 'headtop', 'left_clavicle', 'lshoulder', 'lelbow', 'lwrist', 'left_hand', 'right_clavicle', 'rshoulder', 'relbow', 'rwrist', 'right_hand', 'lhip', 'lknee', 'lankle', 'left_foot', 'left_toe', 'rhip', 'rknee', 'rankle', 'right_foot', 'right_toe']

def get_insta_joint_names():
    return ['OP RHeel', 'OP RKnee', 'OP RHip', 'OP LHip', 'OP LKnee', 'OP LHeel', 'OP RWrist', 'OP RElbow', 'OP RShoulder', 'OP LShoulder', 'OP LElbow', 'OP LWrist', 'OP Neck', 'headtop', 'OP Nose', 'OP LEye', 'OP REye', 'OP LEar', 'OP REar', 'OP LBigToe', 'OP RBigToe', 'OP LSmallToe', 'OP RSmallToe', 'OP LAnkle', 'OP RAnkle']

def get_spin_joint_names():
    return ['OP Nose', 'OP Neck', 'OP RShoulder', 'OP RElbow', 'OP RWrist', 'OP LShoulder', 'OP LElbow', 'OP LWrist', 'OP MidHip', 'OP RHip', 'OP RKnee', 'OP RAnkle', 'OP LHip', 'OP LKnee', 'OP LAnkle', 'OP REye', 'OP LEye', 'OP REar', 'OP LEar', 'OP LBigToe', 'OP LSmallToe', 'OP RBigToe', 'OP RSmallToe', 'OP LHeel', 'OP RHeel', 'rankle', 'rknee', 'rhip', 'lhip', 'lknee', 'lankle', 'rwrist', 'relbow', 'rshoulder', 'lshoulder', 'lelbow', 'lwrist', 'neck', 'headtop', 'hip', 'Spine (H36M)', 'Head (H36M)', 'RShoulder', 'LShoulder', 'RElbow', 'LElbow', 'RWrist', 'LWrist', 'RHip', 'LHip', 'RKnee', 'LKnee', 'RAnkle', 'LAnkle']

def get_common_joint_names():
    return ['rankle', 'rknee', 'rhip', 'lhip', 'lknee', 'lankle', 'rwrist', 'relbow', 'rshoulder', 'lshoulder', 'lelbow', 'lwrist', 'neck', 'headtop']

def get_h36m_joint_names():
    return ['hip', 'lhip', 'lknee', 'lankle', 'rhip', 'rknee', 'rankle', 'Spine (H36M)', 'neck', 'headtop', 'lshoulder', 'lelbow', 'lwrist', 'rshoulder', 'relbow', 'rwrist']

def get_posetrack_joint_names():
    return ['OP Nose', 'OP LEye', 'OP REye', 'OP LEar', 'OP REar', 'OP LShoulder', 'OP RShoulder', 'OP LElbow', 'OP RElbow', 'OP LWrist', 'OP RWrist', 'OP LHip', 'OP RHip', 'OP LKnee', 'OP RKnee', 'OP LAnkle', 'OP RAnkle']

def get_penn_action_joint_names():
    return ['headtop', 'lshoulder', 'rshoulder', 'lelbow', 'relbow', 'lwrist', 'rwrist', 'lhip', 'rhip', 'lknee', 'rknee', 'lankle', 'rankle']
