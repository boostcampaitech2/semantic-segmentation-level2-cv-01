import os
import cv2
import numpy as np
import random

from pycocotools.coco import COCO
from torch.utils.data import Dataset, Subset

class CustomDataset(Dataset):
    """COCO format"""
    def __init__(self, annotation, mode = 'train', transform = None):
        super().__init__()
        self.dataset_path = '/opt/ml/segmentation/input/data/'
        self.mode = mode
        self.transform = transform
        self.coco = COCO(os.path.join(self.dataset_path, annotation))
        
    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]
        
        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(self.dataset_path, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0
        
        if (self.mode in ('train', 'val')):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)

            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id" 할당
            # Background = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            # General trash = 1, ... , Cigarette = 10
            anns = sorted(anns, key=lambda idx : len(idx['segmentation'][0]), reverse=False)
            for i in range(len(anns)):
                # className = get_classname(anns[i]['category_id'], cats)
                # pixel_value = category_names.index(className)
                pixel_value = anns[i]['category_id']
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
    
    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())

def collate_fn(batch):
    return tuple(zip(*batch))