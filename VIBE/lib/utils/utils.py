import os
import yaml
import time
import torch
import shutil
import logging
import operator
from tqdm import tqdm
from os import path as osp
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

    correct = pred.eq(target).sum()

    return correct.item(), target.size(0) - correct.item()


def lr_decay(optimizer, step, lr, decay_step, gamma):
    lr = lr * gamma ** (step/decay_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def step_decay(optimizer, step, lr, decay_step, gamma):
    lr = lr * gamma ** (step / decay_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def read_yaml(filename):
    return yaml.load(open(filename, 'r'))


def write_yaml(filename, object):
    with open(filename, 'w') as f:
        yaml.dump(object, f)


def save_dict_to_yaml(obj, filename, mode='w'):
    with open(filename, mode) as f:
        yaml.dump(obj, f, default_flow_style=False)


def save_to_file(obj, filename, mode='w'):
    with open(filename, mode) as f:
        f.write(obj)


def concatenate_dicts(dict_list, dim=0):
    return {k: torch.cat([d[k] for d in dict_list], dim=dim) for k in dict_list[0].keys()}


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return f'{self.avg:.4f}'


def setup_logger(logger_name, log_file, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def create_logger(log_dir, phase='train'):
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    log_file = os.path.join(log_dir, f'{phase}.log')
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    root_logger.addHandler(fh)
    root_logger.addHandler(ch)

    return root_logger
