# MMSegmentation library baseline
 
MMSegmentation 환경 구축 후 실행 가능합니다.

## Training

    $ python tools/train.py ./deeplab3plus/deeplabv3plus_r50.py

## Inference

    $ python inference.py <config 경로> <checkpoint 경로>

## Baseline Code

### Dataset

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

#### Dataset_small

제대로 돌아가고 있는 지 확인할 수 있는 dataset_small.py 코드도 활용할 수 있습니다. 자세한 사항은 
https://www.notion.so/6708af81f31a47c08833ccb06af293c7?v=7963ba00a3094dedae81aa8f6b03fc80&p=f1f98acae5b34869a129bdb586e3c39e 
을 확인해주세요!

### Runtime

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

### Scheduler

schedule_SGD.py에서 runtime을 수정할 수 있습니다.
lr_config를 

```python
lr_config = dict(
    policy='fixed',
)
```
로 하시면 scheduler 없이 실험하실 수 있습니다.

추가적으로, mmsegmentation의 tools/train.py에 들어가서 argparser의 --deterministic의 default를 True, --seed의 default를 2021로 설정해주시면 seed 고정을 할 수 있습니다!
