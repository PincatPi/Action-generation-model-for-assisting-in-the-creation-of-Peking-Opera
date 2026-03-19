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

class Evaluator():
    def __init__(
            self,
            test_loader,
            model,
            device=None,
    ):
        self.test_loader = test_loader
        self.model = model
        self.device = device

        self.evaluation_accumulators = dict.fromkeys(['pred_j3d', 'target_j3d', 'target_theta', 'pred_verts'])

        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def validate(self):
        self.model.eval()

        start = time.time()

        summary_string = ''

        bar = Bar('Validation', fill='#', max=len(self.test_loader))

        accumulators = dict.fromkeys(self.evaluation_accumulators, [])

        for batch in self.test_loader:
            batch = move_dict_to_device(batch, self.device)

            with torch.no_grad():
                preds = self.model(batch)

            if 'theta' in preds:
                pred_cam = preds['theta'][:, :3]
                pred_pose = preds['theta'][:, 3:75]
                pred_betas = preds['theta'][:, 75:]

            if 'verts' in preds:
                pred_verts = preds['verts']
                accumulators['pred_verts'].append(pred_verts.cpu().numpy())

            if 'kp_3d' in preds:
                pred_j3d = preds['kp_3d']
                accumulators['pred_j3d'].append(pred_j3d.cpu().numpy())

            if 'target_theta' in batch:
                accumulators['target_theta'].append(batch['target_theta'].cpu().numpy())

            if 'target_j3d' in batch:
                accumulators['target_j3d'].append(batch['target_j3d'].cpu().numpy())

            bar.next()

        bar.finish()

        for key in accumulators:
            accumulators[key] = np.concatenate(accumulators[key], axis=0)

        return accumulators
