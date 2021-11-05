# dataset settings
dataset_type = 'CustomDataset'
data_root = '/opt/ml/segmentation/input/mmseg/'

# class settings
classes = ['Background', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic','Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
palette = [
    [0, 0, 0],
    [192, 0, 128], [0, 128, 192], [0, 128, 64],
    [128, 0, 0], [64, 0, 128], [64, 0, 192],
    [192, 128, 64], [192, 192, 128], [64, 64, 128], [128, 0, 192]
    ]

# set normalize value
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512),ratio_range=(0.5, 1.5)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion', contrast_range = (0.8,1.2), saturation_range = (1.0,1.0)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(512,512)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion', contrast_range = (0.8,1.2), saturation_range = (1.0,1.0)),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=2 ,
    train=dict(
        classes=classes,
        palette=palette,
        type=dataset_type,
        reduce_zero_label=False, 
        img_dir=data_root + "images/train_all",
        ann_dir=data_root + "annotations/train_all",
        pipeline=train_pipeline),
    val=dict(
        classes=classes,
        palette=palette,
        type=dataset_type,
        reduce_zero_label=False,
        img_dir=data_root + "images/validation",
        ann_dir=data_root + "annotations/validation",
        pipeline=test_pipeline),
    test=dict(
        classes=classes,
        palette=palette,
        type=dataset_type,
        reduce_zero_label=False,
        img_dir=data_root + "images/test",
        pipeline=test_pipeline))