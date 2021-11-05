import os
import mmcv

import sys
from mmcv import Config
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import single_gpu_test
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel

import pandas as pd
import numpy as np

config_dir = sys.argv[1]
checkpoint_dir = sys.argv[2]
work_dir = os.path.split(config_dir)[0]
sub_dir = os.path.split(checkpoint_dir)[1][:-4]+'.csv'




# config file 들고오기
cfg = Config.fromfile(config_dir)
root='/opt/ml/segmentation/input/mmseg/test'

# dataset config 수정
cfg.data.test.img_dir = root
cfg.data.test.pipeline[1]['img_scale'] = (512,512) # Resize
cfg.data.test.test_mode = True

cfg.data.samples_per_gpu = 4

cfg.work_dir = work_dir

cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
cfg.model.train_cfg = None


dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(
        dataset,
        samples_per_gpu=4,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
checkpoint = load_checkpoint(model, checkpoint_dir, map_location='cpu')

model.CLASSES = dataset.CLASSES
model = MMDataParallel(model.cuda(), device_ids=[0])

output = single_gpu_test(model, data_loader)
output = np.asarray(output)
np.save(os.path.split(checkpoint_dir)[1][:-4], output)
