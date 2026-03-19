#!/usr/bin/env python
import argparse
import os
import sys
import traceback
import time
import pickle
from collections import OrderedDict
import yaml
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class IO():
    def __init__(self, work_dir, save_log=True, print_log=True):
        self.work_dir = work_dir
        self.save_log = save_log
        self.print_to_screen = print_log
        self.cur_time = time.time()
        self.split_timer = {}
        self.session_file = None
        self.model_text = ''

    def load_model(self, model, **model_args):
        Model = import_class(model)
        model = Model(**model_args)
        self.model_text += '\n\n' + str(model)
        return model

    def load_weights(self, model, weights_path, ignore_weights=None, fix_weights=False):
        if ignore_weights is None:
            ignore_weights = []
        if isinstance(ignore_weights, str):
            ignore_weights = [ignore_weights]

        self.print_log(f'Load weights from {weights_path}.')
        weights = torch.load(weights_path)
        weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in weights.items()])

        for i in ignore_weights:
            ignore_name = list()
            for w in weights:
                if w.find(i) == 0:
                    ignore_name.append(w)
            for n in ignore_name:
                weights.pop(n)
                self.print_log(f'Filter [{i}] remove weights [{n}].')

        try:
            model.load_state_dict(weights)
        except (KeyError, RuntimeError):
            state = model.state_dict()
            diff = list(set(state.keys()).difference(set(weights.keys())))
            state.update(weights)
            model.load_state_dict(state)

        return model

    def print_log(self, log_str, print_time=True):
        if print_time:
            log_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + ' ' + log_str
        if self.save_log:
            with open(os.path.join(self.work_dir, 'log.txt'), 'a') as f:
                f.write(log_str + '\n')
        if self.print_to_screen:
            print(log_str)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def str2dict(v):
    result = {}
    for item in v.split(','):
        key, value = item.split(':')
        result[key] = int(value)
    return result


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


class DictAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super(DictAction, self).__init__(option_strings, dest, nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, str2dict(values))
