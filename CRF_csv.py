import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels
import pandas as pd
import numpy as np
import json
import cv2
import os

def decode_pred(pred):
    m = pred.split()
    img = np.zeros(256*256, dtype=np.uint8)
    for i, j in enumerate(m):
        img[i] = int(j)
    return img.reshape(256,256)

def crf(ori_img, mask_img):
    #pdb.set_trace()
    labels = mask_img.flatten()
    n_labels = 11

    d = dcrf.DenseCRF2D(ori_img.shape[1], ori_img.shape[0], n_labels)
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=(3,3), compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    d.addPairwiseBilateral(sxy=(10,10), srgb=(70,70,70), rgbim=ori_img, compat=5)
 
    # crf inference 할 횟수.
    Q = d.inference(40)

    MAP = np.argmax(Q, axis=0)

    return MAP.reshape((ori_img.shape[0], ori_img.shape[1]))

# 변환하고자 하는 csv 파일
df = pd.read_csv('/opt/ml/segmentation/mmsegmentation/work_dirs/fin_upernet_swin_soft/no_test_aug_submission_fin_uper_44epoch.csv')
#root='/opt/ml/segmentation/input/mmseg/test'



json_dir = os.path.join("/opt/ml/segmentation/input/data/test.json")
with open(json_dir, "r", encoding="utf8") as outfile:
    datas = json.load(outfile)

# 새로 csv 파일을 만들기 위해 빈 데이터프레임 생성
df2 = pd.DataFrame(index=range(0,819), columns=['image_id','PredictionString'])

for i in range(819):
    #o_img = cv2.imread(root + str(i).zfill(4) + '.jpg')
    o_img = cv2.imread('/opt/ml/segmentation/input/data/' + datas['images'][i]['file_name'])
    o_img = cv2.resize(o_img, (256,256))
    m_img = decode_pred(df['PredictionString'][i])
    crf_img = crf(o_img, m_img)
    df2['image_id'][i] = datas['images'][i]['file_name']
    df2['PredictionString'][i] = ' '.join(str(e) for e in list(crf_img.flatten()))

# crf로 나온 새로운 데이터프레임을 csv로 변환

df2.to_csv('/opt/ml/segmentation/mmsegmentation/work_dirs/fin_upernet_swin_soft/crf_notestaug_fin_uper_44epoch.csv', index=False)