import time
import torch
import shutil
import logging
import numpy as np
import os.path as osp
from progress.bar import Bar

from lib.core.config import VIBE_DATA_DIR
from lib.utils.utils import move_dict_to_device, AverageMeter

from lib.utils.eval_utils import (
    compute_accel,
    compute_error_accel,
    compute_error_verts,
    batch_compute_similarity_transform_torch,
)

logger = logging.getLogger(__name__)


class Trainer():
    def __init__(
            self,
            data_loaders,
            generator,
            motion_discriminator,
            gen_optimizer,
            dis_motion_optimizer,
            dis_motion_update_steps,
            end_epoch,
            criterion,
            start_epoch=0,
            lr_scheduler=None,
            motion_lr_scheduler=None,
            device=None,
            writer=None,
            debug=False,
            debug_freq=1000,
            logdir='output',
            resume=None,
            performance_type='min',
            num_iters_per_epoch=1000,
    ):

        self.train_2d_loader, self.train_3d_loader, self.disc_motion_loader, self.valid_loader = data_loaders

        self.disc_motion_iter = iter(self.disc_motion_loader)

        self.train_2d_iter = self.train_3d_iter = None

        if self.train_2d_loader:
            self.train_2d_iter = iter(self.train_2d_loader)

        if self.train_3d_loader:
            self.train_3d_iter = iter(self.train_3d_loader)

        self.generator = generator
        self.gen_optimizer = gen_optimizer

        self.motion_discriminator = motion_discriminator
        self.dis_motion_optimizer = dis_motion_optimizer

        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.motion_lr_scheduler = motion_lr_scheduler
        self.device = device
        self.writer = writer
        self.debug = debug
        self.debug_freq = debug_freq
        self.logdir = logdir

        self.dis_motion_update_steps = dis_motion_update_steps

        self.performance_type = performance_type
        self.train_global_step = 0
        self.valid_global_step = 0
        self.epoch = 0
        self.best_performance = float('inf') if performance_type == 'min' else -float('inf')

    def train(self):
        self.generator.train()
        self.motion_discriminator.train()

        for self.epoch in range(self.start_epoch, self.end_epoch):
            self.train_one_epoch()

    def train_one_epoch(self):
        self.generator.train()
        self.motion_discriminator.train()

        bar = Bar('Training', fill='#', max=self.num_iters_per_epoch)

        for i in range(self.num_iters_per_epoch):
            self.train_one_step()

            bar.suffix = f'Epoch: [{self.epoch}/{self.end_epoch}]'
            bar.next()

        bar.finish()

    def train_one_step(self):
        pass

    def validate(self):
        pass

    def save_model(self):
        pass
