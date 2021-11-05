## 사용법
파일 안의 TODO 표시된 부분을 사용하시는 환경에 맞게 수정한 후 실행하시면 됩니다.
<br />
<br />

### TODO List
<br />

```
# TODO
# SMP 모델 지정
model = smp.DeepLabV3Plus(
    encoder_name="efficientnet-b5",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=11,                      # model output channels (number of classes in your dataset)
)
```
- model에 SMP에서 사용하신 모델을 지정해주시면 됩니다.

```
# TODO
# 체크포인트 경로 지정
model_path = '/opt/ml/segmentation/baseline_code/saved/DeepLabV3Plus_efficientnet-b5_best_model.pt'
```
- 모델의 체크포인트 파일의 경로를 지정해줍니다.

```
# TODO
# 저장할 json 파일 이름 지정
pseudo_labeling_name = 'pseudo_labeling_test'
```
- pseudo labeling의 이름을 정해줍니다.  이는 나중에 mmsegmentation dataset의 디렉토리 이름으로도 사용됩니다.


<br />

## 동작 방식

이미 학습된 모델을 불러와 test data에 대해 inference한 뒤
기존에 있던 train.json COCO format 파일에 pseudo labeling한 데이터를 추가한 새로운 json 파일을 만드는 방식입니다.<br />

pseudo labeling의 threshold는 분포를 보고 임의로 정했습니다.  기준은 픽셀마다 점수의 평균입니다.<br />

MMSegmentation 모델 버전도 만들었지만 모델의 예측 결과물이 중간에 점수를 알 수 없고 바로 분류 결과만 나와버려 threshold를 적용할 수 없어 폐기했습니다.
