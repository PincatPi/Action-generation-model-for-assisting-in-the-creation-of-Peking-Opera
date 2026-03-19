import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import torch
import pprint
import random
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from lib.core.loss import VIBELoss
from lib.core.trainer import Trainer
from lib.core.config import parse_args
from lib.utils.utils import prepare_output_dir
from lib.models import VIBE, MotionDiscriminator
from lib.dataset.loaders import get_data_loaders
from lib.utils.utils import create_logger, get_optimizer


def main(cfg):
    if cfg.SEED_VALUE >= 0:
        print(f'Seed value for the experiment {cfg.SEED_VALUE}')
        os.environ['PYTHONHASHSEED'] = str(cfg.SEED_VALUE)
        random.seed(cfg.SEED_VALUE)
        torch.manual_seed(cfg.SEED_VALUE)
        np.random.seed(cfg.SEED_VALUE)

    logger = create_logger(cfg.LOGDIR, phase='train')

    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')

    logger.info(pprint.pformat(cfg))

    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    writer = SummaryWriter(log_dir=cfg.LOGDIR)
    writer.add_text('config', pprint.pformat(cfg), 0)

    data_loaders = get_data_loaders(cfg)

    loss = VIBELoss(
        e_loss_weight=cfg.LOSS.KP_2D_W,
        e_3d_loss_weight=cfg.LOSS.KP_3D_W,
        e_pose_loss_weight=cfg.LOSS.POSE_W,
        e_shape_loss_weight=cfg.LOSS.SHAPE_W,
        d_motion_loss_weight=cfg.LOSS.D_MOTION_LOSS_W,
    )

    generator = VIBE(
        n_layers=cfg.MODEL.TGRU.NUM_LAYERS,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        seqlen=cfg.DATASET.SEQLEN,
        hidden_size=cfg.MODEL.TGRU.HIDDEN_SIZE,
        pretrained=cfg.TRAIN.PRETRAINED_REGRESSOR,
        add_linear=cfg.MODEL.TGRU.ADD_LINEAR,
        bidirectional=cfg.MODEL.TGRU.BIDIRECTIONAL,
        use_residual=cfg.MODEL.TGRU.RESIDUAL,
    ).to(cfg.DEVICE)

    if cfg.TRAIN.PRETRAINED != '' and os.path.isfile(cfg.TRAIN.PRETRAINED):
        checkpoint = torch.load(cfg.TRAIN.PRETRAINED)
        best_performance = checkpoint['performance']
        generator.load_state_dict(checkpoint['gen_state_dict'])
        print(f'==> Loaded pretrained model from {cfg.TRAIN.PRETRAINED}...')
        print(f'Performance on 3DPW test set {best_performance}')
    else:
        print(f'{cfg.TRAIN.PRETRAINED} is not a pretrained model!!!!')

    gen_optimizer = get_optimizer(
        model=generator,
        optim_type=cfg.TRAIN.GEN_OPTIM,
        lr=cfg.TRAIN.GEN_LR,
        weight_decay=cfg.TRAIN.GEN_WD,
        momentum=cfg.TRAIN.GEN_MOMENTUM,
    )

    motion_discriminator = MotionDiscriminator(
        input_size=63,
        hidden_size=cfg.TRAIN.MOT_DISCR.HIDDEN_SIZE,
        num_layers=cfg.TRAIN.MOT_DISCR.NUM_LAYERS,
    ).to(cfg.DEVICE)

    dis_motion_optimizer = get_optimizer(
        model=motion_discriminator,
        optim_type=cfg.TRAIN.MOT_DISCR.OPTIM,
        lr=cfg.TRAIN.MOT_DISCR.LR,
        weight_decay=cfg.TRAIN.MOT_DISCR.WD,
        momentum=cfg.TRAIN.MOT_DISCR.MOMENTUM,
    )

    trainer = Trainer(
        data_loaders=data_loaders,
        generator=generator,
        motion_discriminator=motion_discriminator,
        gen_optimizer=gen_optimizer,
        dis_motion_optimizer=dis_motion_optimizer,
        dis_motion_update_steps=cfg.TRAIN.MOT_DISCR.UPDATE_STEPS,
        end_epoch=cfg.TRAIN.END_EPOCH,
        criterion=loss,
        start_epoch=cfg.TRAIN.START_EPOCH,
        device=cfg.DEVICE,
        writer=writer,
        debug=cfg.DEBUG,
        debug_freq=cfg.DEBUG_FREQ,
        logdir=cfg.LOGDIR,
        resume=cfg.TRAIN.RESUME,
        num_iters_per_epoch=cfg.TRAIN.NUM_ITERS_PER_EPOCH,
    )

    trainer.train()


if __name__ == '__main__':
    cfg, cfg_file = parse_args()
    main(cfg)
