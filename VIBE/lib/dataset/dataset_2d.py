import os
import torch
import random
import logging
import numpy as np
import os.path as osp
import joblib

from torch.utils.data import Dataset

from lib.core.config import VIBE_DB_DIR
from lib.data_utils.kp_utils import convert_kps
from lib.data_utils.img_utils import normalize_2d_kp, transfrom_keypoints, split_into_chunks

logger = logging.getLogger(__name__)

class Dataset2D(Dataset):
    def __init__(self, seqlen, overlap=0., folder=None, dataset_name=None, debug=False):
        self.folder = folder
        self.dataset_name = dataset_name
        self.seqlen = seqlen
        self.stride = int(seqlen * (1-overlap))
        self.debug = debug
        self.db = self.load_db()
        self.vid_indices = split_into_chunks(self.db['vid_name'], self.seqlen, self.stride)

    def __len__(self):
        return len(self.vid_indices)

    def __getitem__(self, index):
        return self.get_single_item(index)

    def load_db(self):
        set = 'train'
        db_file = osp.join(VIBE_DB_DIR, f'{self.dataset_name}_{set}_db.pt')
        if osp.isfile(db_file):
            db = joblib.load(db_file)
        else:
            raise ValueError(f'{db_file} do not exists')
        print(f'Loaded {self.dataset_name} dataset from {db_file}')
        return db

    def get_single_item(self, index):
        start_index, end_index = self.vid_indices[index]
        kp_2d = self.db['joints2D'][start_index:end_index+1]
        if self.dataset_name != 'posetrack':
            kp_2d = convert_kps(kp_2d, src=self.dataset_name, dst='spin')
        kp_2d_tensor = np.ones((self.seqlen, 49, 3), dtype=np.float16)
        bbox = self.db['bbox'][start_index:end_index+1]
        input = torch.from_numpy(self.db['features'][start_index:end_index+1]).float()

        for idx in range(self.seqlen):
            kp_2d[idx,:,:2], trans = transfrom_keypoints(
                kp_2d=kp_2d[idx,:,:2],
                center_x=bbox[idx,0],
                center_y=bbox[idx,1],
                width=bbox[idx,2],
                height=bbox[idx,3],
                patch_width=224,
                patch_height=224,
                do_augment=False,
            )
            kp_2d[idx,:,:2] = normalize_2d_kp(kp_2d[idx,:,:2], 224)
            kp_2d_tensor[idx] = kp_2d[idx]

        vid_name = self.db['vid_name'][start_index:end_index+1]
        frame_id = self.db['img_name'][start_index:end_index+1].astype(str)
        instance_id = np.array([v+f for v,f in zip(vid_name, frame_id)])

        target = {
            'features': input,
            'kp_2d': torch.from_numpy(kp_2d_tensor).float(),
            'instance_id': instance_id,
        }
        return target
