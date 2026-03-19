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

    if len(tracking_results) == 0:
        print('No tracks found in the video')
        return

    model = VIBE_Demo(
        seqlen=16,
        n_layers=2,
        hidden_size=1024,
        pretrained=args.ckpt,
    ).to(device)

    model.eval()

    for person_id in tqdm(tracking_results.keys()):
        print(f'Processing person id {person_id}')
        bboxes = tracking_results[person_id]['bbox']
        frames = tracking_results[person_id]['frames']

        dataset = Inference(
            image_folder=image_folder,
            frames=frames,
            bboxes=bboxes,
            scale=bbox_scale,
        )

        batch_size = args.batch_size
        data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=8)

        with torch.no_grad():
            pred_cam, orig_cam, verts, pose, betas, joints3d = [], [], [], [], [], []
            for batch in data_loader:
                batch = batch.unsqueeze(0)
                batch = batch.to(device)
                batch = batch.reshape(-1, 16, 224, 224)
                output = model(batch)[-1]
                pred_cam.append(output['pred_cam'])
                verts.append(output['verts'])
                pose.append(output['pose'])
                betas.append(output['betas'])

    print(f'Total time: {time.time() - total_time}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vid_file', type=str, help='input video path or youtube url')
    parser.add_argument('--output_folder', type=str, default='output', help='output folder')
    parser.add_argument('--ckpt', type=str, default='data/vibe_data/vibe_model_w_3dpw.pth.tar', help='checkpoint path')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--display', action='store_true', help='display results')
    parser.add_argument('--tracking_method', type=str, default='mpt', choices=['mpt', 'pose'], help='tracking method')
    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', 'maskrcnn'], help='detector type')
    parser.add_argument('--yolo_img_size', type=int, default=416, help='yolo image size')
    parser.add_argument('--tracker_batch_size', type=int, default=12, help='tracker batch size')
    args = parser.parse_args()

    main(args)
