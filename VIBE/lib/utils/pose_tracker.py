import os
import json
import shutil
import subprocess
import numpy as np
import os.path as osp


def run_openpose(
        video_file,
        output_folder,
        staf_folder,
        vis=False,
):
    pwd = os.getcwd()

    os.chdir(staf_folder)

    render = 1 if vis else 0
    display = 2 if vis else 0
    cmd = [
        'build/examples/openpose/openpose.bin',
        '--model_pose', 'BODY_21A',
        '--tracking', '1',
        '--render_pose', str(render),
        '--video', video_file,
        '--write_json', output_folder,
        '--display', str(display)
    ]

    print('Executing', ' '.join(cmd))
    subprocess.call(cmd)
    os.chdir(pwd)


def run_posetracker(video_file, output_folder, staf_folder, vis=False):
    run_openpose(video_file, output_folder, staf_folder, vis)

    json_files = sorted([f for f in os.listdir(output_folder) if f.endswith('.json')])

    keypoints = []
    for json_file in json_files:
        with open(osp.join(output_folder, json_file), 'r') as f:
            data = json.load(f)

        if len(data['people']) == 0:
            keypoints.append(np.zeros((21, 3)))
        else:
            keypoints.append(np.array(data['people'][0]['pose_keypoints_2d']).reshape(-1, 3)[:21])

    return np.array(keypoints)
