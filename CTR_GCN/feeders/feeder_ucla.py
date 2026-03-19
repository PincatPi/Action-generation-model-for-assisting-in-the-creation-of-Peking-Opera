import numpy as np
import json

from torch.utils.data import Dataset

class Feeder(Dataset):
    def __init__(self, data_path, label_path, repeat=1, random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True):

        self.nw_ucla_root = 'data/NW-UCLA/all_sqe/'
        self.time_steps = 52
        self.bone = [(1, 2), (2, 3), (3, 3), (4, 3), (5, 3), (6, 5), (7, 6), (8, 7), (9, 3), (10, 9), (11, 10),
                     (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19)]

        if 'val' in label_path:
            self.train_val = 'val'
            self.data_dict = []
        else:
            self.train_val = 'train'
            self.data_dict = []

        self.label = []
        for index in range(len(self.data_dict)):
            info = self.data_dict[index]
            self.label.append(int(info['label']) - 1)

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.repeat = repeat
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        self.data = []
        for data in self.data_dict:
            file_name = data['file_name']
            with open(self.nw_ucla_root + file_name + '.json', 'r') as f:
                json_file = json.load(f)
            skeletons = json_file['skeletons']
            value = np.array(skeletons)
            self.data.append(value)

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label) * self.repeat

    def __getitem__(self, index):
        index = index % len(self.label)
        data_numpy = self.data[index]
        label = self.label[index]
        return data_numpy, label, index
