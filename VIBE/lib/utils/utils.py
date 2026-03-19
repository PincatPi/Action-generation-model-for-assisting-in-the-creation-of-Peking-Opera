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
    out_dict = dict_list[0].copy()
    for i in range(1, len(dict_list)):
        for k, v in dict_list[i].items():
            out_dict[k] = torch.cat([out_dict[k], v], dim=dim)
    return out_dict


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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
