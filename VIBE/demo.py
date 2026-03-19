import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import cv2
import time
import torch
import joblib
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from multi_person_tracker import MPT
from torch.utils.data import DataLoader

from lib.models.vibe import VIBE_Demo
from lib.utils.renderer import Renderer
from lib.dataset.inference import Inference
from lib.utils.smooth_pose import smooth_pose
from lib.data_utils.kp_utils import convert_kps
from lib.utils.pose_tracker import run_posetracker

from lib.utils.demo_utils import (
    download_youtube_clip,
    smplify_runner,
    convert_crop_coords_to_orig_img,
    convert_crop_cam_to_orig_img,
    prepare_rendering_results,
    video_to_images,
    images_to_video,
    download_ckpt,
)

MIN_NUM_FRAMES = 25


def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    video_file = args.vid_file

    if video_file.startswith('https://www.youtube.com'):
        print(f'Donwloading YouTube video "{video_file}"')
        video_file = download_youtube_clip(video_file, '/tmp')

        if video_file is None:
            exit('Youtube url is not valid!')

        print(f'YouTube Video has been downloaded to {video_file}...')

    if not os.path.isfile(video_file):
        exit(f'Input video "{video_file}" does not exist!')

    output_path = os.path.join(args.output_folder, os.path.basename(video_file).replace('.mp4', ''))
    os.makedirs(output_path, exist_ok=True)

    image_folder, num_frames, img_shape = video_to_images(video_file, return_info=True)

    print(f'Input video number of frames {num_frames}')
    orig_height, orig_width = img_shape[:2]

    total_time = time.time()

    bbox_scale = 1.1
    if args.tracking_method == 'pose':
        if not os.path.isabs(video_file):
            video_file = os.path.join(os.getcwd(), video_file)
        tracking_results = run_posetracker(video_file, staf_folder=args.staf_dir, display=args.display)
    else:
        mot = MPT(
            device=device,
            batch_size=args.tracker_batch_size,
            display=args.display,
            detector_type=args.detector,
            output_format='dict',
            yolo_img_size=args.yolo_img_size,
        )
        tracking_results = mot(image_folder)

    for person_id in list(tracking_results.keys()):
        if tracking_results[person_id]['frames'].shape[0] < MIN_NUM_FRAMES:
            del tracking_results[person_id]

    model = VIBE_Demo(
        seqlen=16,
        n_layers=2,
        hidden_size=1024,
        add_linear=True,
        use_residual=True,
    ).to(device)

    pretrained_file = download_ckpt(use_3dpw=False)
    ckpt = torch.load(pretrained_file)
    print(f'Performance of pretrained model on 3DPW: {ckpt["performance"]}')
    ckpt = ckpt['gen_state_dict']
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    print(f'Loaded pretrained weights from "{pretrained_file}"')

    print(f'Running VIBE on each tracklet...')
    vibe_time = time.time()
    vibe_results = {}
    for person_id in tqdm(list(tracking_results.keys())):
        bboxes = joints2d = None

        if args.tracking_method == 'bbox':
            bboxes = tracking_results[person_id]['bbox']
        elif args.tracking_method == 'pose':
            joints2d = tracking_results[person_id]['joints2d']

        frames = tracking_results[person_id]['frames']

        dataset = Inference(
            image_folder=image_folder,
            frames=frames,
            bboxes=bboxes,
            joints2d=joints2d,
            scale=bbox_scale,
        )

        bboxes = dataset.bboxes
        frames = dataset.frames
        has_keypoints = True if joints2d is not None else False

        dataloader = DataLoader(dataset, batch_size=args.vibe_batch_size, num_workers=16)

        with torch.no_grad():
            pred_cam, orig_cam, verts, pose, betas, joints3d = [], [], [], [], [], []

            for batch in dataloader:
                batch = batch.unsqueeze(0)
                batch = batch.to(device)

                predictions = model(batch)

                for pred in predictions:
                    pred_cam.append(pred['pred_cam'])
                    verts.append(pred['verts'])
                    pose.append(pred['pose'])
                    betas.append(pred['betas'])
                    joints3d.append(pred['kp_3d'])

            pred_cam = torch.cat(pred_cam, dim=0)
            verts = torch.cat(verts, dim=0)
            pose = torch.cat(pose, dim=0)
            betas = torch.cat(betas, dim=0)
            joints3d = torch.cat(joints3d, dim=0)

        if args.smooth:
            verts, pose, joints3d = smooth_pose(pose.cpu().numpy(), betas.cpu().numpy())

        vibe_results[person_id] = {
            'pred_cam': pred_cam.cpu().numpy(),
            'orig_cam': orig_cam,
            'verts': verts,
            'pose': pose,
            'betas': betas.cpu().numpy(),
            'joints3d': joints3d,
            'bboxes': bboxes,
            'frame_ids': frames,
        }

    print(f'VIBE completed in {time.time() - vibe_time:.2f} seconds')

    shutil.rmtree(image_folder)

    return vibe_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vid_file', type=str)
    parser.add_argument('--output_folder', type=str, default='output')
    parser.add_argument('--tracking_method', type=str, default='bbox', choices=['bbox', 'pose'])
    parser.add_argument('--detector', type=str, default='yolo', choices=['maskrcnn', 'yolo'])
    parser.add_argument('--yolo_img_size', type=int, default=416)
    parser.add_argument('--tracker_batch_size', type=int, default=12)
    parser.add_argument('--vibe_batch_size', type=int, default=450)
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--smooth', action='store_true')
    parser.add_argument('--staf_dir', type=str, default='/path/to/staf')
    args = parser.parse_args()

    main(args)
