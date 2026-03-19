import argparse
from yacs.config import CfgNode as CN

VIBE_DB_DIR = 'data/vibe_db'
AMASS_DIR = 'data/amass'
INSTA_DIR = 'data/insta_variety'
MPII3D_DIR = 'data/mpi_inf_3dhp'
THREEDPW_DIR = 'data/3dpw'
PENNACTION_DIR = 'data/penn_action'
POSETRACK_DIR = 'data/posetrack'
VIBE_DATA_DIR = 'data/vibe_data'

cfg = CN()

cfg.OUTPUT_DIR = 'results'
cfg.EXP_NAME = 'default'
cfg.DEVICE = 'cuda'
cfg.DEBUG = True
cfg.LOGDIR = ''
cfg.NUM_WORKERS = 8
cfg.DEBUG_FREQ = 1000
cfg.SEED_VALUE = -1

cfg.CUDNN = CN()
cfg.CUDNN.BENCHMARK = True
cfg.CUDNN.DETERMINISTIC = False
cfg.CUDNN.ENABLED = True

cfg.TRAIN = CN()
cfg.TRAIN.DATASETS_2D = ['Insta']
cfg.TRAIN.DATASETS_3D = ['MPII3D']
cfg.TRAIN.DATASET_EVAL = 'ThreeDPW'
cfg.TRAIN.BATCH_SIZE = 32
cfg.TRAIN.DATA_2D_RATIO = 0.5
cfg.TRAIN.START_EPOCH = 0
cfg.TRAIN.END_EPOCH = 5
cfg.TRAIN.PRETRAINED_REGRESSOR = ''
cfg.TRAIN.PRETRAINED = ''
cfg.TRAIN.RESUME = ''
cfg.TRAIN.NUM_ITERS_PER_EPOCH = 1000
cfg.TRAIN.LR_PATIENCE = 5

cfg.TRAIN.GEN_OPTIM = 'Adam'
cfg.TRAIN.GEN_LR = 1e-4
cfg.TRAIN.GEN_WD = 1e-4
cfg.TRAIN.GEN_MOMENTUM = 0.9

cfg.TRAIN.MOT_DISCR = CN()
cfg.TRAIN.MOT_DISCR.OPTIM = 'SGD'
cfg.TRAIN.MOT_DISCR.LR = 1e-2
cfg.TRAIN.MOT_DISCR.WD = 1e-4
cfg.TRAIN.MOT_DISCR.MOMENTUM = 0.9
cfg.TRAIN.MOT_DISCR.UPDATE_STEPS = 1
cfg.TRAIN.MOT_DISCR.FEATURE_POOL = 'concat'
cfg.TRAIN.MOT_DISCR.HIDDEN_SIZE = 1024
cfg.TRAIN.MOT_DISCR.NUM_LAYERS = 1

cfg.DATASET = CN()
cfg.DATASET.SEQLEN = 20
cfg.DATASET.OVERLAP = 0.5

cfg.LOSS = CN()
cfg.LOSS.KP_2D_W = 60.
cfg.LOSS.KP_3D_W = 30.
cfg.LOSS.SHAPE_W = 0.001
ncfg.LOSS.POSE_W = 1.0
cfg.LOSS.D_MOTION_LOSS_W = 1.

cfg.MODEL = CN()
cfg.MODEL.TEMPORAL_TYPE = 'gru'

cfg.MODEL.TGRU = CN()
cfg.MODEL.TGRU.NUM_LAYERS = 1

def get_config():
    return cfg.clone()
