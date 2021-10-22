# Segmentation Models PyTorch(SMP) Baseline
SMP library를 이용한 Segmentation Baseline입니다.<br/>
https://github.com/qubvel/segmentation_models.pytorch

-------
# 사용법
첨부된 utils.py / custom.py / smp_Unet2plus.py 세 script 파일을 같은 디렉터리에 저장하시고 터미널에서 실행하시면 됩니다. 실행은 다음과 같이 하실 수 있습니다.
```
python smp_Unet2plus.py # 아무 설정도 하지 않을 때
python smp_Unet2plus.py --debug # debugging 모드
python smp_Unet2plus.py --deterministic # seed 고정하여 재현성 확보
python smp_Unet2plus.py --private # 개인 wandb entity에 실험 기록
python smp_Unet2plus.py --debug --deterministic --private # 중복해서 사용 가능
```

# 참고
모델마다 적용 가능한 backbone, weight 등이 정리되어 있습니다.<br/>
https://smp.readthedocs.io/en/latest/encoders.html<br/>
Timmm Encoders 탭을 참고하시면 timm 라이브러리의 더 다양한 Backbone을 사용하실 수 있습니다.

-------
# 설명
### utils.py
베이스라인 코드에 있는 것과 같은 파일입니다. label_accuracy_score 메소드에서 클래스별 정확도를 추가로 return하기 위해 아래와 같이 수정했습니다.
```
acc = np.diag(hist).sum() / hist.sum()
with np.errstate(divide='ignore', invalid='ignore'):
    acc_cls = np.diag(hist) / hist.sum(axis=1)
# acc_cls = np.nanmean(acc_cls)
mean_acc_cls = np.nanmean(acc_cls)

'''
'''

# return acc, acc_cls, mean_iu, fwavacc, iu
return acc, acc_cls, mean_acc_cls, mean_iu, fwavacc, iu
```
<br/>

### custom.py
베이스라인에 제공된 CustomDataLoader 클래스와 collate_fn 함수를 포함합니다.

- 모드 별로 데이터셋을 더 간단하게 생성하기 위해 init 함수를 조금 수정했습니다.
```
def __init__(self, annotation, mode = 'train', transform = None):
    super().__init__()
    self.dataset_path = '/opt/ml/segmentation/input/data/'
    self.mode = mode
    self.transform = transform
    self.coco = COCO(os.path.join(self.dataset_path, annotation))
```
<br/>

- mask의 pixel_value를 구하기 위해 불필요하게 category id를 구하는 부분을 수정했습니다.
```
def __getitem__(self, index: int):

	'''
	'''

	for i in range(len(anns)):
	    # className = get_classname(anns[i]['category_id'], cats)
	    # pixel_value = category_names.index(className)
	    pixel_value = anns[i]['category_id']
	    masks[self.coco.annToMask(anns[i]) == 1] = pixel_value
	masks = masks.astype(np.int8)
```
<br/>

- ustomDataLoader 클래스의 이름을 CustomDatset으로 변경했고, 디버깅 모드 사용에 따라 데이터셋을 작게 나눠줄 수 있도록 함수를 추가했습니다.
입력 ratio에 따라 데이터셋을 해당 비율만큼 나누어 작은 데이터셋을 사용할 수 있습니다.
split_dataset() 함수는 smp_Unet2plus.py의 make_dataloader() 함수에서 호출됩니다.
```
def split_dataset(self, ratio=0.1):
    """
    Split dataset into small dataset for debugging.

    Args:
        ratio (float) : Ratio of dataset to use for debugging
            (default : 0.1)

    Returns:
        Subset (obj : Dataset) : Splitted small dataset
    """
    num_data = len(self)
    num_sub_data = int(num_data * ratio)
    indices = list(range(num_data))
    sub_indices = random.choices(indices, k=num_sub_data)
    return Subset(self, sub_indices)
```

### smp_train.py
모델 학습을 위한 스크립트 파일입니다.

- 터미널에서 실행하면서 학습 모드를 결정하기 위해 parse_args() 함수를 추가했습니다.
- make_dataloader() 함수는 train / validation / test 각 모드에 맞게, 그리고 debugging 여부에 따라 dataloader를 생성하는 함수입니다. Augmentation을 적용하실 경우 이 함수안에서 정의하시고 사용하시면 될 것 같습니다.
- set_random_seed() 시드를 고정하는 함수입니다.
- save_model() : 모델을 저장하는 함수입니다. 모델 저장은 loss와 mIoU가 모두 best일 때의 epoch을 저장하도록 했습니다.
- train() : 실제 학습을 수행하는 함수입니다.
- validation() : 한 에폭 학습 후 validation을 수행하는 함수입니다.
- main : model, loss, optimizer 등 실험을 위한 변인과 Hyperparameter를 정의하는 공간입니다.

자세한 설명은 docstring으로 적어놨으니 참고하시면 될 것 같습니다. 그리고 실험에 따라 수정이 필요한 부분은 찾기 쉽게 ## TODO 주석을 달아놨으니 검색해서 찾으시면 될 것 같습니다.

### smp_inference.ipynb
모델 제출을 위한 노트북 파일입니다. 베이스라인에서 필요한 부분만 가져왔습니다.
inference도 나중에 스크립트로 바꿔서 터미널에서 실행할 수 있도록 변경하겠습니다.

-------------
# Version
### Version 1
- Version 1.1 : U-Net2++ baseline
- Version 1.2
	- 카테고리별로 train_IoU, train_acc, valid_IoU, valid_acc 기록하도록 수정
	- 디버깅 모드에서는 모델 저장하지 않도록 수정
	- 모델 저장 시 saved_dir에 모델 이름으로 하위 디렉터리 만들고, 모델 이름을 사용하여 모델 저장할 수 있도록 수정
	- 모델 저장 기준을 mIoU만 적용, save_interval 적용하여 n번째 epoch마다 모델 저장하도록 수정
	- 학습 종료 후 마지막 모델 저장하도록 수정
	- 디버깅 모드에서 wandb run name 앞에 debug_ 붙여서 관리하기 쉽도록 수정
- Version 1.3
    	- wandb에 Module, hyperparameter를 기록하도록 수정
    	- mIoU에 따라 best model을 저장 / save_interval마다 best mIoU를 기록한 epoch의 모델 저장 / 학습 종료 후 마지막 모델 저장하도록 수정
- Version 1.4
	- make_save_dir() 함수 추가 - 학습 시작 시 새로운 실험의 checkpoint를 저장하기 위한 디렉터리 생성
	- 같은 모델과 백본으로 다른 실험을 하는 경우 모델 저장 시 checkpoint가 변경되는 문제가 있어서 실험이 중복되지 않게 새로운 디렉터리에 저장하도록 수정
- Version 1.5
	- get_model_name() 함수 추가 - DeepLab v3+처럼 model에 name attribute가 없는 경우, model.name을 사용하면서 발생한 에러 수정
	- debugging 모드에서는 모델 생성 / 디렉터리 생성을 하지 않도록 수정
	- train / validation data loader 생성 시 drop_last=True로 설정하여 에러 수정
	- wandb와 모델 저장 디렉터리에 backbone 이름이 추가되지 않는 문제 수정 (임시 수정)
