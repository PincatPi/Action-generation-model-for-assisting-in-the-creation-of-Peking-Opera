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

class Dataset3D(Dataset):
    def __init__(self, set, seqlen, overlap=0., folder=None, dataset_name=None, debug=False):
        self.folder = folder
        self.set = set
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
        db_file = osp.join(VIBE_DB_DIR, f'{self.dataset_name}_{self.set}_db.pt')
        if osp.isfile(db_file):
            db = joblib.load(db_file)
        else:
            raise ValueError(f'{db_file} do not exists')
        print(f'Loaded {self.dataset_name} dataset from {db_file}')
        return db

    def get_single_item(self, index):
        start_index, end_index = self.vid_indices[index]
        is_train = self.set == 'train'

        if self.dataset_name == '3dpw':
            kp_2d = convert_kps(self.db['joints2D'][start_index:end_index + 1], src='common', dst='spin')
            kp_3d = self.db['joints3D'][start_index:end_index + 1]
        elif self.dataset_name == 'mpii3d':
            kp_2d = self.db['joints2D'][start_index:end_index + 1]
            if is_train:
                kp_3d = self.db['joints3D'][start_index:end_index + 1]
            else:
                kp_3d = convert_kps(self.db['joints3D'][start_index:end_index + 1], src='spin', dst='common')
        elif self.dataset_name == 'h36m':
            kp_2d = self.db['joints2D'][start_index:end_index + 1]
            if is_train:
                kp_3d = self.db['joints3D'][start_index:end_index + 1]
            else:
                kp_3d = convert_kps(self.db['joints3D'][start_index:end_index + 1], src='spin', dst='common')

        kp_2d_tensor = np.ones((self.seqlen, 49, 3), dtype=np.float16)
        nj = 14 if not is_train else 49
        kp_3d_tensor = np.zeros((self.seqlen, nj, 3), dtype=np.float16)

        if self.dataset_name == '3dpw':
            pose = self.db['pose'][start_index:end_index+1]
            shape = self.db['shape'][start_index:end_index+1]
            w_smpl = torch.ones(self.seqlen).float()
            w_3d = torch.ones(self.seqlen).float()
        elif self.dataset_name == 'h36m':
            if not is_train:
                pose = np.zeros((kp_2d.shape[0], 72))
                shape = np.zeros((kp_2d.shape[0], 10))
                w_smpl = torch.zeros(self.seqlen).float()
                w_3d = torch.ones(self.seqlen).float()
            else:
                pose = self.db['pose'][start_index:end_index + 1]
                shape = self.db['shape'][start_index:end_index + 1]

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
            kp_3d_tensor[idx] = kp_3d[idx]

        target = {
            'features': input,
            'kp_2d': torch.from_numpy(kp_2d_tensor).float(),
            'kp_3d': torch.from_numpy(kp_3d_tensor).float(),
            'pose': torch.from_numpy(pose).float(),
            'shape': torch.from_numpy(shape).float(),
            'w_smpl': w_smpl,
            'w_3d': w_3d,
        }
        return target
