import os
import yaml
import torch
import logging
from tqdm import tqdm
from functools import reduce
from typing import List, Union


def move_dict_to_device(dict, device, tensor2float=False):
    for k,v in dict.items():
        if isinstance(v, torch.Tensor):
            if tensor2float:
                dict[k] = v.float().to(device)
            else:
                dict[k] = v.to(device)


def get_from_dict(dict, keys):
    return reduce(operator.getitem, keys, dict)


def tqdm_enumerate(iter):
    i = 0
    for y in tqdm(iter):
        yield i, y
        i += 1


def iterdict(d):
    for k,v in d.items():
        if isinstance(v, dict):
            d[k] = dict(v)
            iterdict(v)
    return d


def accuracy(output, target):
    _, pred = output.topk(1)
    pred = pred.view(-1)
    correct = pred.eq(target)
    correct = correct.view(-1).float().sum(0)
    return correct.item()


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, path):
    with open(path, 'w') as f:
        yaml.dump(config, f)
