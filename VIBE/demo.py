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
        video_file = download_youtube_clip(video_file)

    if os.path.isfile(video_file):
        video_path = video_file
    else:
        print(f'Input video "{video_file}" does not exist')
        exit(0)

    image_folder, num_frames, img_shape = video_to_images(video_path)
    print(f'Input video file is {video_file}')
    print(f'Number of frames {num_frames}')

    model = VIBE_Demo(seqlen=16).to(device)
    model_file = os.path.join(VIBE_DATA_DIR, 'vibe_model_w_3dpw.pth.tar')
    model.load_state_dict(torch.load(model_file))
    model.eval()

    dataset = Inference(
        image_folder=image_folder,
        frames=None,
        bboxes=None,
        kp2d=None,
    )

    inference_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    print('Running VIBE inference...')
    with torch.no_grad():
        for batch in tqdm(inference_loader):
            batch = move_dict_to_device(batch, device)
            output = model(batch)

    print('Saving output...')
    output_file = os.path.join(args.output_folder, 'vibe_output.pkl')
    joblib.dump(output, output_file)

    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vid_file', type=str, help='input video path or youtube link')
    parser.add_argument('--output_folder', type=str, default='output', help='output folder to write results')
    parser.add_argument('--display', action='store_true', help='visualize the results')
    args = parser.parse_args()

    main(args)
