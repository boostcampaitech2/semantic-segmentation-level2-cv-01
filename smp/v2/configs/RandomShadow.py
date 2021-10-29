val_every = 1
save_interval = 3
batch_size = 8   # Mini-batch size
num_epochs = 50
learning_rate = 0.0001
exp_num = 12

TRAIN_LEN = 2617

classes = ['Backgroud', 'General trash', 'Paper', 'Paper pack', 'Metal',
    'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
saved_dir = '/opt/ml/segmentation/saved'

model = dict(
    type='DeepLabV3Plus',
    args = dict(
        encoder_name='resnet50',
        in_channels=3,
        classes=11
    )
)

train_pipeline=[
    dict(
        type='HorizontalFlip',
        args=dict(p=0.5)
    ),
    dict(
        type='VerticalFlip',
        args=dict(p=0.5)
    ),
    dict(
        type='RandomRotate90',
        args=dict(p=0.5)
    ),
    dict(
        type='GridDropout',
        args=dict(
            ratio=0.3, holes_number_x=5, holes_number_y=5,
            shift_x=100, shift_y=100, random_offset=True,
            fill_value=0, always_apply=False, p=0.5
        )
    ),
    dict(
        type='RandomBrightnessContrast',
        args=dict(
            brightness_limit=0.2,
            contrast_limit=0.2,
            brightness_by_max=False,
            always_apply=False, p=0.5
        )
    ),
    dict(
        type='RandomShadow',
        args=dict(
            shadow_roi=(0, 0.5, 1, 1),
            num_shadows_lower=1,
            num_shadows_upper=2,
            shadow_dimension=5,
            always_apply=False,
            p=0.5
        )
    ),
    dict(
        type='ToTensorV2'
    )
]

test_pipeline=[
    dict(
        type='ToTensorV2'
    )
]


data = dict(
    train=dict(
        type='train',
        annotation='train.json',
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        ratio=0.1,
        pipeline=train_pipeline
    ),
    valid=dict(
        type='valid',
        annotation='val.json',
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        ratio=0.1,
        pipeline=test_pipeline
    ),
    test=dict(
        type='test',
        annotation='test.json',
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        ratio=0.1,
        pipeline=test_pipeline
    )
)

criterion = dict(
    type='CrossEntropyLoss'
)

optimizer = dict(
    type='Adam',
    args=dict(
        lr = learning_rate,
        weight_decay=1e-5
        )
)

scheduler = dict(
    type='CosineAnnealingLR',
    args=dict(
        T_max=(TRAIN_LEN // batch_size) * num_epochs // 5
    )
)

wandb = dict(
    entity = 'bagineer',
    team_entity = 'perforated_line',
    project = 'AugTeam_NA'
)