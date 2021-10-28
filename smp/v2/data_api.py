import os
import cv2
import numpy as np
import random
import albumentations as A
import albumentations.pytorch as AP

from pycocotools.coco import COCO
from torch.utils.data import Dataset, Subset, DataLoader

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
        
        if (self.mode in ('train', 'valid')):
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

def get_transforms(pipeline):
    _transforms = []
    for _transform in pipeline:
        if isinstance(_transform, dict):
            if hasattr(A, _transform.type):
                transform = getattr(A, _transform.type)
                if hasattr(_transform, 'args'):
                    transform = transform(**_transform.args)
                _transforms.append(transform)
            elif _transform.type == 'ToTensorV2':
                transform = getattr(AP, _transform.type)
                _transforms.append(transform())
            else:
                raise KeyError(f"albumentations has no module named '{_transform.type}'.")
        elif isinstance(_transform, list):
            _transforms.append(get_transforms(_transform))
        else:
            raise TypeError(f"{pipeline} is not type of (dict, list).")

    transforms = A.Compose(_transforms)
    return transforms

def build_loader(cfg_data, debug=False):

    """
    Create dataloader by arguments.

    Args:
        mode (str) : Type of dataset (default : 'train')
            e.g. mode='train', mode='val', mode='test'
        
        batch_size (int) : Batch size (default : 8)

        suffle (bool) : Whether to shuffle dataset when creating loader
            (default : False)
        
        num_workers (int) : Number of processors (default : 4)
        
        collate_fn (func) : Collate function for Dataset
            (default : collate_fn from custom)

        ratio (float) : Ratio of splited Dataset
        
        debug (bool) : Debugging mode (default : False)

    Returns:
        loader (obj : DataLoader) : DataLoader created by arguments
    """

    annotation = cfg_data.annotation
    transforms = get_transforms(cfg_data.pipeline)
    drop_last = cfg_data.type in ['train', 'val']

    dataset = CustomDataset(annotation=annotation, mode=cfg_data.type, transform=transforms)
    if debug:
        dataset = dataset.split_dataset(ratio=cfg_data.ratio)
    loader = DataLoader(dataset=dataset,
                        batch_size=cfg_data.batch_size,
                        shuffle=cfg_data.shuffle,
                        num_workers=cfg_data.num_workers,
                        collate_fn=collate_fn,
                        drop_last=drop_last)
    
    return loader