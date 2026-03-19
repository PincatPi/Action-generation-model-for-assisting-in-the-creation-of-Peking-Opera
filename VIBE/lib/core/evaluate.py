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

        if self.evaluation_accumulators is not None:
            for k,v in self.evaluation_accumulators.items():
                self.evaluation_accumulators[k] = []

        J_regressor = torch.from_numpy(np.load(osp.join(VIBE_DATA_DIR, 'J_regressor_h36m.npy'))).float()

        for i, target in enumerate(self.test_loader):
            move_dict_to_device(target, self.device)

            with torch.no_grad():
                inp = target['features']
                preds = self.model(inp, J_regressor=J_regressor)

            bar.suffix = f'[{i}/{len(self.test_loader)}]'
            bar.next()

        bar.finish()

        return summary_string

    def run(self):
        self.validate()
