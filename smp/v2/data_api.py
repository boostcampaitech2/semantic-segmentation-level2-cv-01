import os
import cv2
import numpy as np
import random
import pandas as pd
import albumentations as A
import albumentations.pytorch as AP

from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset, Subset, DataLoader

class BalancedAugDataset(Dataset):
    """
    This Dataset is for custom augmentation applied to few classes.
    """
    def __init__(self, annotation, mode = 'train', transform = None):
        super().__init__()
        self.dataset_path = '/opt/ml/segmentation/input/data/'
        self.mode = mode
        self.transform = transform
        self.coco = COCO(os.path.join(self.dataset_path, annotation))
        self.priority = [10, 1, 6, 4, 3, 5]
        self.weight = np.array([0.4595, 0.2983, 0.1013, 0.0748, 0.0387, 0.0275])
        self.weight /= self.weight.sum()
        self.img_list = pd.read_csv('/opt/ml/segmentation/input/data/img_list.csv')
        
    def __getitem__(self, index: int):
        
        target_id = self._get_target_id()
        
        if self.mode == 'train':
            images, masks, image_infos = self._random_resize_and_mix(index, target_id)
            masks = masks.astype(np.int8)
                        
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
            
            images /= 255.0
            return images, masks, image_infos
        elif self.mode == 'val':
            images, masks, image_infos = self._get_items(index)
            
            return images, masks, image_infos
        
        elif self.mode == 'test':
            # dataset이 index되어 list처럼 동작
            image_id = self.coco.getImgIds(imgIds=index)
            image_infos = self.coco.loadImgs(image_id)[0]

            # cv2 를 활용하여 image 불러오기
            images = cv2.imread(os.path.join(self.dataset_path, image_infos['file_name']))
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
            images /= 255.0
            
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

    def _get_items(self, index):
        """
        Ordinary function to get data.
        """
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]
        
        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(self.dataset_path, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0

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
    
    def _get_target_id(self):
        """
        Get id of the target image to crop the patch of an object.
        self.priority is a list of an order of few classes.
        First, Select class id from self.priority with weight (probabilities to be chosen).
        And randomly select one of image ids that have selected class object.
        
        Returns:
            target_id (int) : Selected target image id.
        """
        cat_id = np.random.choice(
            self.priority,
            p=self.weight
        )
        target_id = random.choice(self.img_list.loc[cat_id-1, 'img_ids'].split())
        
        return int(target_id)
    
    def _get_data(self, idx):
        """
        Get data by index from annotations.

        Args:
            idx (int): Index for data
        
        Returns:
            info (obj): Image data that matches index.

            image (obj): Image that matches index.

            anns (obj): Annoataion data that matches index.
        """
        image_id = self.coco.getImgIds(imgIds=idx)
        info = self.coco.loadImgs(image_id)[0]

        image = Image.open(os.path.join(self.dataset_path, info['file_name']))
        image = np.array(image)
        
        ann_ids = self.coco.getAnnIds(imgIds=info['id'])
        anns = self.coco.loadAnns(ann_ids)

        return info, image, anns

    def _get_annotation(self, anns):
        """
        Get an annotation from annotations list of an image.
        First, sort annotations list by area in ascending order.
        And search object that matches class priority order and has area over than 32*32.
        Return annotation data if something had benn searched.
        If not, return annotation data that has the largest area.

        Args:
            anns (obj): Annotation data.
        
        Return:
            anns (obj): Searched annotation data.
        """
        anns = sorted(anns, key=lambda idx : idx['area'], reverse=False)
        
        for cat_id in self.priority:
            for ann in anns:
                if ann['category_id'] == cat_id and ann['area'] >= 32*32:
                    return ann

        return anns[-1]

    def _get_mask(self, info, ann):
        """
        Get mask from annotation data.

        Args:
            info (obj): Image data for image size.

            ann (obj): Annotation data to get mask.

        Returns:
            mask (obj): Mask image of the annotation as numpy array.

            cat_id (int): Category id of the annotation.
        """
        mask = np.zeros((info["height"], info["width"]))

        mask[self.coco.annToMask(ann) == 1] = 1
        cat_id = ann['category_id']

        return mask, cat_id

    def _crop_image_and_mask(self, ann, image, mask):
        """
        Crop image and mask from target image.

        Args:
            ann (obj): Annotation data of target image.

            image (obj): Target image to be cropped.

            mask (obj): Target mask to be cropped.

        Returns:
            cropped_image (obj): Cropped target image as numpy array.
            
            cropped_mask (obj): Cropped target mask as numpy array.
        """
        bbox = np.round(ann['bbox'])

        ltx = int(bbox[1])
        lty = int(bbox[0])
        rbx = int(bbox[1] + bbox[3])
        rby = int(bbox[0] + bbox[2])

        cropped_image = image[ltx:rbx, lty:rby, :]
        cropped_mask = mask[ltx:rbx, lty:rby]

        return cropped_image, cropped_mask

    def _random_resize_image_and_mask(self, image, mask):
        """
        Randomly resize target image and target mask.

        Args:
            image (obj): Cropped target image to be resized.

            mask (obj): Cropped target mask to be resized.

        Returns:
            resized_image (obj): Resized target image.
            
            resized_mask (obj): Resized target mask.
        """
        scale = np.random.randint(low=(512//4), high=(512//3))

        resized_image = np.zeros(image.shape)
        w, h, c = image.shape
        ratio = scale/w if w > h else scale/h

        new_w, new_h = int(w*ratio), int(h*ratio)

        resized_image = cv2.resize(image,
                           dsize=(new_h, new_w),
                           interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(mask,
                          dsize=(new_h, new_w),
                          interpolation=cv2.INTER_LINEAR)

        return resized_image, resized_mask
    
    def _random_rotate_flip(self, image, mask):
        """
        Randomly rotate and flip image.
        Rotation options are
        [clockwise 90 degrees, counter-clockwise 90 degrees, 180 degrees, 0 degree].
        Flip options are
        [horizontal, vertical, horizontal and vertical, None].
        Probabilities of all options are same.

        Args:
            image (obj): Image to be flipped and rotated.

            mask (obj): Mask to be flipped and rotated.

        Returns:
            flipped_image (obj): Transformed image. 
            
            flipped_mask (obj): Transformed mask.

        """
        ROTATE = [cv2.ROTATE_90_CLOCKWISE,
                  cv2.ROTATE_90_COUNTERCLOCKWISE,
                  cv2.ROTATE_180,
                  None]
        FLIP = [0, 1, -1, None]
        
        rotated_image = image
        rotated_mask = mask
        
        rotate_id = np.random.randint(low=0, high=4)
        if ROTATE[rotate_id]:
            rotated_image = cv2.rotate(rotated_image, ROTATE[rotate_id])
            rotated_mask = cv2.rotate(rotated_mask, ROTATE[rotate_id])
            
        flipped_image = rotated_image
        flipped_mask = rotated_mask
            
        flip_id = np.random.randint(low=0, high=4)
        if FLIP[flip_id]:
            flipped_image = cv2.flip(flipped_image, FLIP[flip_id])
            flipped_mask = cv2.flip(flipped_mask, FLIP[flip_id])
            
        return flipped_image, flipped_mask

    def _random_shift_image_and_mask(self, info, image, mask):
        """
        Randomly Shift image and mask.
        Left top corner coordinates of the image and mask are to be close to
        one-third of origin image height and width.

        Args:
            info (obj): Image data for image size.

            image (obj): Image to be shifted.

            mask (obj): Mask to be shifted.

        Returns:
            shifted_image (obj): Shifted image.

            shifted_mask (obj): Shifted mask.
            
        """
        WIDTH, HEIGHT = info["width"], info["height"]
        w, h, c = image.shape

        x_s = np.random.randint(low=0, high=WIDTH//3)
        y_s = np.random.randint(low=0, high=HEIGHT//3)
        x_l = np.random.randint(low=WIDTH//3*2, high=WIDTH + 1)
        y_l = np.random.randint(low=HEIGHT//3*2, high=HEIGHT + 1)

        x_r = np.random.randint(low=0, high=WIDTH+1-w)
        y_r = np.random.randint(low=0, high=HEIGHT+1-h)

        if x_s + w < WIDTH:
            x = random.choice((x_s, x_l)) if x_l + w < WIDTH else x_s
        else:
            x = x_l if x_l + w < WIDTH else x_r

        if y_s + h < HEIGHT:
            y = random.choice((y_s, y_l)) if y_l + h < HEIGHT else y_s
        else:
            y = y_l if y_l + h < HEIGHT else y_r

        shifted_image = np.zeros((WIDTH, HEIGHT, 3), dtype=int)
        shifted_mask = np.zeros((WIDTH, HEIGHT), dtype=int)
        shifted_image[x:x+w, y:y+h, :] = image
        shifted_mask[x:x+w, y:y+h] = mask

        return shifted_image.astype(np.uint8), shifted_mask.astype(np.uint8)

    def _merge_mask(self, origin_info, origin_anns, mask, cat_id):
        """
        Merge original mask and target mask.

        Args:
            origin_info (obj): Image data of original image.

            origin_anns (obj): Annotation data of original image.

            mask (obj): Target mask.

            cat_id (int): Category of target mask.

        Returns:
            merged_mask (obj): Mask that merged original mask and target mask.

        """
        origin_mask = np.zeros((origin_info["height"], origin_info["width"]))
        origin_anns = sorted(origin_anns, key=lambda idx : idx['area'], reverse=False)
        for i in range(len(origin_anns)):
            pixel_value = origin_anns[i]['category_id']
            origin_mask[self.coco.annToMask(origin_anns[i]) == 1] = pixel_value
            
        merged_mask = np.where(mask==0, origin_mask, cat_id)

        return merged_mask

    def _random_resize_and_mix(self, origin_id, target_id):
        """
        This is a custom augmentation.
        Crop image of target id and resize, rotate, flip, shift it randomly.
        And mix transfromed target image with original image also mask.
        If there's no valid target annotation data from _get_data function,
        return original image, mask, info.
        This function is called in the __getitem__ function.

        Args:
            origin_id (int): Original image id

            target_id (int): Target image id

        Returns:
            image (obj): Transformed image if there's valid target annotation,
                original image if not.

            mask (obj): Transformed mask if there's valid target annotation,
                original image if not.

            info (obj): Original image data

        """
        origin_info, origin_image, origin_anns = self._get_data(origin_id)
        target_info, target_image, target_anns = self._get_data(target_id)

        if not target_anns:
            origin_masks = np.zeros((origin_info["height"], origin_info["width"]))
            origin_anns = sorted(origin_anns, key=lambda idx : idx['area'], reverse=False)
            for i in range(len(origin_anns)):
                pixel_value = origin_anns[i]['category_id']
                origin_masks[self.coco.annToMask(origin_anns[i]) == 1] = pixel_value
            origin_masks = origin_masks.astype(np.int8)
            return origin_image.astype(np.float32), origin_masks, origin_info

        target_ann = self._get_annotation(target_anns)
        mask, cat_id = self._get_mask(target_info, target_ann)

        for i in range(3):
            target_image[:, :, i] = target_image[:, :, i] * mask

        target_image, mask = self._crop_image_and_mask(target_ann, target_image, mask)    
        target_image, mask = self._random_resize_image_and_mask(target_image, mask)   
        target_image, mask = self._random_rotate_flip(target_image, mask)
        target_image, mask = self._random_shift_image_and_mask(target_info, target_image, mask)

        masked_image = np.where(target_image==0, origin_image , target_image)
        merged_mask = self._merge_mask(origin_info, origin_anns, mask, cat_id)

        return masked_image.astype(np.float32), merged_mask, origin_info

class CustomAugDataset(Dataset):
    """
    This Dataset is for custom augmentation.
    """
    def __init__(self, annotation, mode = 'train', transform = None):
        super().__init__()
        self.dataset_path = '/opt/ml/segmentation/input/data/'
        self.mode = mode
        self.transform = transform
        self.coco = COCO(os.path.join(self.dataset_path, annotation))
        
    def __getitem__(self, index: int):
        max_id = len(self)
        target_id = random.choice(list(set(range(max_id)) - set([index])))
        
        if (self.mode in ('train', 'val')):
            images, masks, image_infos = self.random_resize_and_mix(index, target_id)
            masks = masks.astype(np.int8)
                        
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
            
            images /= 255.0
            return images, masks, image_infos
        
        if self.mode == 'test':
            # dataset이 index되어 list처럼 동작
            image_id = self.coco.getImgIds(imgIds=index)
            image_infos = self.coco.loadImgs(image_id)[0]

            # cv2 를 활용하여 image 불러오기
            images = cv2.imread(os.path.join(self.dataset_path, image_infos['file_name']))
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
            images /= 255.0
            
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
    
    def get_info_image(self, idx):
        image_id = self.coco.getImgIds(imgIds=idx)
        info = self.coco.loadImgs(image_id)[0]

        image = Image.open(os.path.join(self.dataset_path, info['file_name']))
        image = np.array(image)

        return info, image

    def get_annotations(self, info):
        ann_ids = self.coco.getAnnIds(imgIds=info['id'])
        anns = self.coco.loadAnns(ann_ids)

        return anns

    def get_mask(self, info, anns):
        mask = np.zeros((info["height"], info["width"]))

        anns = sorted(anns, key=lambda idx : idx['area'], reverse=False)
        mask[self.coco.annToMask(anns[-1]) == 1] = 1
        cat_id = anns[-1]['category_id']

        return mask, cat_id

    def crop_image_and_mask(self, anns, image, mask):
        anns = sorted(anns, key=lambda idx : idx['area'], reverse=False)
        bbox = np.round(anns[-1]['bbox'])

        ltx = int(bbox[1])
        lty = int(bbox[0])
        rbx = int(bbox[1] + bbox[3])
        rby = int(bbox[0] + bbox[2])

        cropped_image = image[ltx:rbx, lty:rby, :]
        cropped_mask = mask[ltx:rbx, lty:rby]

        return cropped_image, cropped_mask

    def random_resize_image_and_mask(self, image, mask):
        scale = np.random.randint(low=(512//4), high=(512//3))

        resized_image = np.zeros(image.shape)
        w, h, c = image.shape
        ratio = scale/w if w > h else scale/h

        new_w, new_h = int(w*ratio), int(h*ratio)

        resized_image = cv2.resize(image,
                           dsize=(new_h, new_w),
                           interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(mask,
                          dsize=(new_h, new_w),
                          interpolation=cv2.INTER_LINEAR)

        return resized_image, resized_mask

    def random_shift_image_and_mask(self, info, image, mask):
        WIDTH, HEIGHT = info["width"], info["height"]
        w, h, c = image.shape

        x_s = np.random.randint(low=0, high=WIDTH//3)
        y_s = np.random.randint(low=0, high=HEIGHT//3)
        x_l = np.random.randint(low=WIDTH//3*2, high=WIDTH + 1)
        y_l = np.random.randint(low=HEIGHT//3*2, high=HEIGHT + 1)

        x_r = np.random.randint(low=0, high=WIDTH+1-w)
        y_r = np.random.randint(low=0, high=HEIGHT+1-h)

        if x_s + w < WIDTH:
            x = random.choice((x_s, x_l)) if x_l + w < WIDTH else x_s
        else:
            x = x_l if x_l + w < WIDTH else x_r

        if y_s + h < HEIGHT:
            y = random.choice((y_s, y_l)) if y_l + h < HEIGHT else y_s
        else:
            y = y_l if y_l + h < HEIGHT else y_r

        shifted_image = np.zeros((WIDTH, HEIGHT, 3), dtype=int)
        shifted_mask = np.zeros((WIDTH, HEIGHT), dtype=int)
        shifted_image[x:x+w, y:y+h, :] = image
        shifted_mask[x:x+w, y:y+h] = mask

        return shifted_image.astype(np.uint8), shifted_mask.astype(np.uint8)

    def merge_mask(self, origin_info, origin_anns, mask, cat_id):
        origin_mask = np.zeros((origin_info["height"], origin_info["width"]))
        origin_anns = sorted(origin_anns, key=lambda idx : idx['area'], reverse=False)
        for i in range(len(origin_anns)):
            pixel_value = origin_anns[i]['category_id']
            origin_mask[self.coco.annToMask(origin_anns[i]) == 1] = pixel_value
            
        merged_mask = np.where(mask==0, origin_mask, cat_id)

        return merged_mask

    def random_resize_and_mix(self, origin_id, target_id):
        origin_info, origin_image = self.get_info_image(origin_id)
        target_info, target_image = self.get_info_image(target_id)
        origin_anns = self.get_annotations(origin_info)
        target_anns = self.get_annotations(target_info)
        mask, cat_id = self.get_mask(target_info, target_anns)

        for i in range(3):
            target_image[:, :, i] = target_image[:, :, i] * mask

        target_image, mask = self.crop_image_and_mask(target_anns, target_image, mask)    
        target_image, mask = self.random_resize_image_and_mask(target_image, mask)
        target_image, mask = self.random_shift_image_and_mask(target_info, target_image, mask)

        masked_image = np.where(target_image==0, origin_image , target_image)
        merged_mask = self.merge_mask(origin_info, origin_anns, mask, cat_id)

        return masked_image.astype(np.float32), merged_mask, origin_info

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