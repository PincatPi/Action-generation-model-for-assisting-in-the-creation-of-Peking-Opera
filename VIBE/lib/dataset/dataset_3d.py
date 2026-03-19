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
from lib.data_utils.img_utils import split_into_chunks

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

        target = {
            'features': torch.from_numpy(self.db['features'][start_index:end_index + 1]).float(),
            'kp_2d': torch.from_numpy(kp_2d).float(),
            'kp_3d': torch.from_numpy(kp_3d).float(),
        }

        return target
