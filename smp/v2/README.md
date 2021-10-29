# SMP Baseline Version 2
## Usage
### Settings
```
!pip install easydict
!pip insatll tqdm
```

### Execute
```
python train.py ${your_config_file.py} --deterministic --private --debug
```
\[--deterministic, --private, --debug\]는 이전과 동일하게 옵션으로 설정하셔서 사용하시면 됩니다.

## Config.py
### hyperparameters
```
val_every = 1   # validaion 주기
save_interval = 3   # 모델 저장 주기
batch_size = 8   # Mini-batch size
num_epochs = 30   # epoch 수
learning_rate = 0.0001  # optimizer learning rate
exp_num = 99    # 실험 번호

classes = ['Backgroud', 'General trash', 'Paper', 'Paper pack', 'Metal',
    'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']    # class
saved_dir = '/opt/ml/segmentation/saved'   # 모델 저장 경로
```

### models
type에 smp segmentation 모델 이름 적으시고 args에 dict type으로 필요한 argument 입력하시면 됩니다.
```
# 모델 정의
model = dict(
    type='DeepLabV3Plus',   # segmentation 모델 이름
    args = dict(
        encoder_name='resnet50',  # encoder name
        in_channels=3,
        classes=11
    )
)
```

### pipelines
albumentations 라이브러리의 augmentation들을 선택해서 넣으시면 됩니다.</br>
마찬가지로 type에 augmentation 이름 적으시고 args에 dict type으로 필요한 값들 적으시면 됩니다. 
```
# pipeline 정의 (Augmentations of albumentations)
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
        type='ToTensorV2'
    )
]

test_pipeline=[
    dict(
        type='ToTensorV2'
    )
]
```

### data
data를 정의하는 부분입니다. loader까지 같이 고려하는 부분이라 loader 생성에 필요한 인자도 같이 넣으시면 됩니다.
```
# 데이터 정의
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
```

### Loss
loss 정의하시면 됩니다. type에 loss function 이름, 필요하시면 다른 것들처럼 args에 dict type으로 추가하시면 될 것 같습니다.
```
# loss 정의
criterion = dict(
    type='CrossEntropyLoss'
)
```

### Optimizer
optimizer 정의도 마찬가지입니다. type에 optimizer 이름, args에 필요한 인자들 넣으시면 됩니다.
```
# optimizer 정의
optimizer = dict(
    type='Adam',
    args=dict(
        lr = learning_rate,
        weight_decay=1e-5
        )
)
```

### Scheduler
scheduler도 마찬가지로 정의하시면 되는데, 사용 안하실거면 지우셔도 될 것 같습니다.
```
# scheduler 정의
scheduler = dict(
    type='CosineAnnealingLR',
    args=dict(
        T_max=300
    )
)
```

### wandb init setting
wandb에 기록할 것들을 정의하시면 됩니다. run_name은 option입니다. 사용하실거면 추가하셔도 될 것 같아요!
```
# wandb config 정의
wandb = dict(
    entity = 'bagineer',
    team_entity = 'perforated_line',
    project = 'AugTeam_NA',
    # run_name = 'optional'
)
```

## Code
### train.py
 - model training을 시작하기 위한 코드입니다. 초기 설정 등이 포함되어 있습니다.

### train_api.py
 - 실제 train과 validation을 하는 코드입니다. 이전과 동일한 구조를 갖고 있습니다.

### config_api.py
 - Config : config 파일로부터 argument를 받아 dictionary 형태로 반환하는 class입니다.

### data_api.py
 - CustomDataset : 이전과 동일합니다.
 - get_transforms : config에 정의된 pipeline을 transform module으로 반환합니다.
 - build_loader : data 타입에 따라 lodaer를 생성합니다. 이전과 동일합니다.

### init_api.py
 - 초기 설정을 위한 코드입니다.
 - make_save_dir : 모델 저장을 위한 directory를 생성합니다.
 - set_random_seed : 시드를 고정합니다. 이전과 동일합니다.
 - set_exp_name : 실험 이름을 설정합니다.
 - init_wandb : wandb에 초기 설정을 기록합니다.

### metric_api.py
 - metric을 계산하기 위한 Metric class가 정의되어 있습니다.</br>
mean_acc, 클래스 별 [acc, IoU], mIoU, mean_loss, mean_fwavacc 등을 계산합니다.

### model_api.py
 - model 및 module 생성과 저장 위한 코드입니다.
 - build_model : 모델을 생성합니다.
 - build_module : optimizer, loss, scheduler 등의 module을 생성합니다.
 - save_model : 모델을 저장합니다. 이전과 동일합니다.

### wandb_api.py
 - wandb에 기록하기 위한 코드입니다.
 - make_wandb_images : wandb에 기록할 이미지를 생성합니다.
 - log_lr : optimizer의 learning rate를 기록합니다.
 - log_train_wandb : train 결과를 wandb에 기록합니다.
 - log_valid_wandb : validation 결과를 wandb에 기록합니다.

## Version
### Version 2.1
 - config 파일로 pipeline을 구축할 수 있도록 개선
 - configs 폴더 안에 config 파일 따로 관리해서 실행 가능하도록 수정
 - best epoch 기록하지 않는 문제 수정
 - learning rate을 step마다 기록하도록 수정
 - config 파일에 train set 길이 추가해서 scheduler 주기 조정
