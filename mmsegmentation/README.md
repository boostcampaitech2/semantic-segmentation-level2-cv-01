# MMSegmentation library baseline
 
MMSegmentation 환경 구축 후 실행 가능합니다.

## Configs

### Training

    $ python tools/train.py ./configs/ocrnet_hr48/ocrnet_hr48.py

### Inference

대회에 알맞는 형식의 csv파일을 생성할 수 있습니다.

    $ python inference.py ./configs/ocrnet_hr48/ocrnet_hr48.py <checkpoint 경로>

### Inference Softvoting

SoftVoting이 가능할 수 있도록 directory에 (819,11,512,512) size의 numpy array를 저장할 수 있습니다.

    $ python inference_soft.py ./configs/ocrnet_hr48/ocrnet_hr48_soft.py <checkpoint 경로>

### Baseline Code

#### Dataset

dataset.py를 보시면,

```python
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512)),
    dict(type='RandomFlip', prob=0.),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', prob=0.),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
```

이 train_pipeline과 test_pipeline을 통해 augmentation을 넣을 수 있습니다. test_pipeline은 validation과 test에 모두 사용됩니다.

samples_per_gpu를 통해 batch_size를 조절할 수 있습니다.

#### Runtime

default_runtime.py의 

```python
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
				# wandb 추가
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='MMSegmentation'
               ))
    ])
```
project의 이름을 수정하여 원하시는 project 이름을 설정할 수 있습니다!

#### Scheduler

schedule_SGD.py에서 runtime을 수정할 수 있습니다.
lr_config를 

```python
lr_config = dict(
    policy='fixed',
)
```
로 하시면 scheduler 없이 실험하실 수 있습니다.

추가적으로, mmsegmentation의 tools/train.py에 들어가서 argparser의 --deterministic의 default를 True, --seed의 default를 2021로 설정해주시면 seed 고정을 할 수 있습니다!

## Data

### EDA

base로 주어진 train_all, train, 그리고 validation set의 분포의 확인을 dataset_EDA.ipynb에서 할 수 있습니다.

### Configuration

redist와 dataset_mmseg 파일을 통해 기존의 dataset을 mmsegmentation이 실행 가능하게 설정할 수 있습니다.

### Loss Weight

Loss의 class weight에 활용할 weight를 주어진 dataset을 통해 구하는 과정을 loss_weight.ipynb에서 확인할 수 있습니다.

## Models

### Focal Loss

Multi-class Focal Loss를 직접 구현했습니다. 

### Soft Voting

Softvoting을 위해 만들어진 segmentor로, .npy 파일을 저장할 수 있도록 일부 수정했습니다.

각 python 파일들을 알맞은 directory에 넣은 후 아래의 command를 실행하시면 됩니다.

    $ pip install -e .

## Final Submission

1. DPT
2. SETR
3. UperNet

