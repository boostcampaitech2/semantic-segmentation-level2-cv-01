import argparse
import os
import random
import time
import json
import warnings 
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import label_accuracy_score, add_hist
import cv2

import numpy as np
import pandas as pd
from tqdm import tqdm

# 전처리를 위한 라이브러리
from pycocotools.coco import COCO
import torchvision
import torchvision.transforms as transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2

# 시각화를 위한 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from matplotlib.patches import Patch
import webcolors
import segmentation_models_pytorch as smp

# TTA를 위한 라이브러리
import ttach as tta



def parse_args():
    """
    Parse arguments from terminal. Use Args in terminel when execute
    this script file.
    e.g. python smp_Unet2plus.py --debug --deterministic --private

    Args:
        optional:
            tta : Apply Test Time augmentation
                (default : False)
    """
    parser = argparse.ArgumentParser(description='Test Segmentation Model')
    parser.add_argument('--tta',
        action='store_true',
        help='whether to apply tta for inference')
    
    parser.add_argument('--model_name',
        help='model to apply from smp : deeplab, u_net, pa_net')
    
    parser.add_argument('--backbone',
        help='backbone to apply from smp : resnet50, resnet101, efficient_b3, efficient_b4, efficient_b5, efficient_b6, efficient_b7')

    args = parser.parse_args()

    return args


class CustomDataLoader(Dataset):
    """COCO format"""
    def __init__(self, data_dir, mode = 'train', transform = None):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.coco = COCO(data_dir)
        
    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]
        
        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(dataset_path, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0
        
        if (self.mode in ('train', 'val')):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)

            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)
            
            print(cat_ids)

            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id" 할당
            # Background = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            # General trash = 1, ... , Cigarette = 10
            anns = sorted(anns, key=lambda idx : len(idx['segmentation'][0]), reverse=False)
            for i in range(len(anns)):
                className = get_classname(anns[i]['category_id'], cats)
                pixel_value = category_names.index(className)
                masks[self.coco.annToMask(anns[i]) == 1] = pixel_value
            masks = masks.astype(np.int8)
                        
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
            return images, masks, image_infos
        
        if self.mode == 'test':
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            return images, image_infos
    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())

def collate_fn(batch):
    return tuple(zip(*batch))

def load_model(model_name, backbone, saved_dir, model_path):
    if model_name == 'deeplab':
        model = smp.DeepLabV3Plus(
            encoder_name=backbone,
            #encoder_weights='noisy-student',
            in_channels=3,
            classes=11
        )
    elif model_name == 'u_net':
        model = smp.UnetPlusPlus(
            encoder_name=backbone,
            #encoder_weights='noisy-student',
            in_channels=3,
            classes=11
        )

    checkpoint = torch.load(os.path.join(saved_dir, model_path), map_location=device)
    state_dict = checkpoint['net']
    model.load_state_dict(state_dict)

    transforms = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.VerticalFlip(),
        tta.Rotate90(angles=[0, 180]),
        tta.Multiply(factors=[0.9, 1, 1.1]),        
    ]
    )
    if tta_mode == True:
        tta_model = tta.SegmentationTTAWrapper(model, transforms)
        return tta_model
    else:
        return model


def test(model, data_loader, device):
    size = 256
    transform = A.Compose([A.Resize(size, size)])
    print('Start prediction.')
    
    model.eval()
    
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)
    
    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(tqdm(test_loader)):
            
            # inference (512 x 512)
            outs = model(torch.stack(imgs).to(device))
            oms = torch.argmax(outs, dim=1).detach().cpu().numpy()
            
            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)
                
            oms = np.array(temp_mask)
            
            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))
            
            file_name_list.append([i['file_name'] for i in image_infos])
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]
    
    return file_names, preds_array


if __name__=='__main__':
    print("="*30 + '\n')
    print('pytorch version: {}'.format(torch.__version__))
    print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))

    print(torch.cuda.get_device_name(0))
    print(torch.cuda.device_count())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = parse_args()
    tta_mode = args.tta
    model_name = args.model_name
    backbone = args.backbone

    dataset_path  = '../../../input/data'
    test_path = dataset_path + '/test.json'
    test_transform = A.Compose([
                           ToTensorV2()
                           ])
    test_dataset = CustomDataLoader(data_dir=test_path, mode='test', transform=test_transform)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=4,
                                            num_workers=4,
                                            collate_fn=collate_fn)
    # TODO
    saved_dir = '/opt/ml/segmentation/semantic-segmentation-level2-cv-01-1/smp/aug_v1/saved'
    model_path = 'DeepLabV3Plus-resnet50_exp5/DeepLabV3Plus-resnet50_best_basic.pt'
                                  
    model = load_model(model_name=model_name, backbone=backbone, saved_dir=saved_dir, model_path=model_path)
    model = model.to(device)

    file_names, preds = test(model, test_loader, device)
    for file_name, string in zip(file_names, preds):
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                    ignore_index=True)
    # TODO
    submission.to_csv(f"./interence1.csv", index=False)