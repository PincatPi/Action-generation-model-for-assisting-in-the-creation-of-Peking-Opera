import h5py
import torch
import logging
import numpy as np
import os.path as osp

from torch.utils.data import Dataset
from lib.core.config import VIBE_DB_DIR
from lib.data_utils.kp_utils import convert_kps
from lib.data_utils.img_utils import split_into_chunks

logger = logging.getLogger(__name__)

class Insta(Dataset):
    def __init__(self, seqlen, overlap=0., debug=False):
        self.seqlen = seqlen
        self.stride = int(seqlen * (1-overlap))

        self.h5_file = osp.join(VIBE_DB_DIR, 'insta_train_db.h5')

        with h5py.File(self.h5_file, 'r') as db:
            self.db = db
            self.vid_indices = split_into_chunks(self.db['vid_name'], self.seqlen, self.stride)

        print(f'InstaVariety number of dataset objects {self.__len__()}')

    def __len__(self):
        return len(self.vid_indices)

    def __getitem__(self, index):
        return self.get_single_item(index)

    def get_single_item(self, index):
        start_index, end_index = self.vid_indices[index]

        with h5py.File(self.h5_file, 'r') as db:
            features = db['features'][start_index:end_index + 1]
            kp_2d = db['joints2D'][start_index:end_index + 1]

        kp_2d = convert_kps(kp_2d, src='insta', dst='spin')

        target = {
            'features': torch.from_numpy(features).float(),
            'kp_2d': torch.from_numpy(kp_2d).float(),
        }

        return target
