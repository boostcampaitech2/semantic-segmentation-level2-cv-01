dataset_type = 'CustomDataset'
data_root = '/opt/ml/segmentation/input/mmseg/'
classes = [
    'Background', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
    'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing'
]
palette = [[0, 0, 0], [192, 0, 128], [0, 128, 192], [0, 128, 64], [128, 0, 0],
           [64, 0, 128], [64, 0, 192], [192, 128, 64], [192, 192, 128],
           [64, 64, 128], [128, 0, 192]]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 1.5)),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PhotoMetricDistortion',
        contrast_range=(0.8, 1.2),
        saturation_range=(1.0, 1.0)),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(512, 512)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='PhotoMetricDistortion',
                contrast_range=(0.8, 1.2),
                saturation_range=(1.0, 1.0)),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=2,
    train=dict(
        classes=[
            'Background', 'General trash', 'Paper', 'Paper pack', 'Metal',
            'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery',
            'Clothing'
        ],
        palette=[[0, 0, 0], [192, 0, 128], [0, 128, 192], [0, 128, 64],
                 [128, 0, 0], [64, 0, 128], [64, 0, 192], [192, 128, 64],
                 [192, 192, 128], [64, 64, 128], [128, 0, 192]],
        type='CustomDataset',
        reduce_zero_label=False,
        img_dir='/opt/ml/segmentation/input/mmseg/images/train_all',
        ann_dir='/opt/ml/segmentation/input/mmseg/annotations/train_all',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 1.5)),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='PhotoMetricDistortion',
                contrast_range=(0.8, 1.2),
                saturation_range=(1.0, 1.0)),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        classes=[
            'Background', 'General trash', 'Paper', 'Paper pack', 'Metal',
            'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery',
            'Clothing'
        ],
        palette=[[0, 0, 0], [192, 0, 128], [0, 128, 192], [0, 128, 64],
                 [128, 0, 0], [64, 0, 128], [64, 0, 192], [192, 128, 64],
                 [192, 192, 128], [64, 64, 128], [128, 0, 192]],
        type='CustomDataset',
        reduce_zero_label=False,
        img_dir='/opt/ml/segmentation/input/mmseg/images/validation',
        ann_dir='/opt/ml/segmentation/input/mmseg/annotations/validation',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=[(512, 512)],
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip', prob=0.5),
                    dict(
                        type='PhotoMetricDistortion',
                        contrast_range=(0.8, 1.2),
                        saturation_range=(1.0, 1.0)),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        classes=[
            'Background', 'General trash', 'Paper', 'Paper pack', 'Metal',
            'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery',
            'Clothing'
        ],
        palette=[[0, 0, 0], [192, 0, 128], [0, 128, 192], [0, 128, 64],
                 [128, 0, 0], [64, 0, 128], [64, 0, 192], [192, 128, 64],
                 [192, 192, 128], [64, 64, 128], [128, 0, 192]],
        type='CustomDataset',
        reduce_zero_label=False,
        img_dir='/opt/ml/segmentation/input/mmseg/images/test',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=[(512,512),(768,768),(1024,1024),(1280,1280)],
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip', prob=0.5),
                    dict(
                        type='PhotoMetricDistortion',
                        contrast_range=(0.8, 1.2),
                        saturation_range=(1.0, 1.0)),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook', init_kwargs=dict(project='RealFinal'))
    ])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.01,
    min_lr=1e-06)
runner = dict(type='EpochBasedRunner', max_epochs=51)
checkpoint_config = dict(interval=10)
evaluation = dict(interval=1, metric='mIoU', save_best='mIoU')
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='CascadeEncoderDecoderSoft',
    num_stages=2,
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        type='HRNet',
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(48, 96)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384)))),
    decode_head=[
        dict(
            type='FCNHead',
            in_channels=[48, 96, 192, 384],
            channels=720,
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            kernel_size=1,
            num_convs=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            concat_input=False,
            dropout_ratio=-1,
            num_classes=11,
            align_corners=False,
            loss_decode=dict(
                type='FocalLoss',
                alpha=[6.4, 6.8, 6.5, 7.7, 7.5, 7.6, 6.7, 7.0, 6.4, 29.1, 8.3],
                loss_name='loss_focal',
                loss_weight=0.4)),
        dict(
            type='OCRHead',
            in_channels=[48, 96, 192, 384],
            channels=512,
            ocr_channels=256,
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            dropout_ratio=-1,
            num_classes=11,
            align_corners=False,
            loss_decode=dict(
                type='FocalLoss',
                alpha=[6.4, 6.8, 6.5, 7.7, 7.5, 7.6, 6.7, 7.0, 6.4, 29.1, 8.3],
                loss_name='loss_focal',
                loss_weight=1.0))
    ],
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
work_dir = './work_dirs/ocrnet_hr48'
gpu_ids = range(0, 1)
