_base_ = [
    './baseline/dataset_TTA.py',
    './baseline/default_runtime.py',
    './baseline/schedule_SGD.py'
]

backbone_norm_cfg = dict(type='LN', eps=1e-06, requires_grad=True)
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    pretrained=
    '/opt/ml/segmentation/mmsegmentation/pretrained/jx_vit_large_p16_384-b3be5167.pth',
    backbone=dict(
        type='VisionTransformer',
        img_size=(512, 512),
        patch_size=16,
        in_channels=3,
        embed_dims=1024,
        num_layers=24,
        num_heads=16,
        out_indices=(9, 14, 19, 23),
        drop_rate=0.0,
        norm_cfg=dict(type='LN', eps=1e-06, requires_grad=True),
        with_cls_token=True,
        interpolate_mode='bilinear'),
    decode_head=dict(
        type='SETRUPHead',
        in_channels=1024,
        channels=256,
        in_index=3,
        num_classes=11,
        dropout_ratio=0,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        num_convs=4,
        up_scale=2,
        kernel_size=3,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=[
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=256,
            in_index=0,
            num_classes=11,
            dropout_ratio=0,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            act_cfg=dict(type='ReLU'),
            num_convs=2,
            kernel_size=3,
            align_corners=False,
            loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=256,
            in_index=1,
            num_classes=11,
            dropout_ratio=0,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            act_cfg=dict(type='ReLU'),
            num_convs=2,
            kernel_size=3,
            align_corners=False,
            loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=256,
            in_index=2,
            num_classes=11,
            dropout_ratio=0,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            act_cfg=dict(type='ReLU'),
            num_convs=2,
            kernel_size=3,
            align_corners=False,
            loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    ],
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)))

cudnn_benchmark = True

optimizer = dict(
    _delete_=True,
    type='SGD',
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0,
    paramwise_cfg=dict(custom_keys=dict(head=dict(lr_mult=10.0))))

lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.1,
    min_lr_ratio=1e-05)
    
work_dir = './work_dirs/setr_pup_512x512_160k_b16_ade20k'
gpu_ids = range(0, 1)