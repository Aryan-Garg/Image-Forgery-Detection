import os

import numpy as np
import cv2 as cv

import torchvision.transforms as T
from torch.utils.data import Dataset


# class Spoof_ImageFolders():
#     def __init__(self, base_dir: str, train_transform=T.Compose([T.ToTensor()]), test_transform=T.Compose([T.ToTensor()])):
#         self.trainImageFolder = ImageFolder(root=base_dir + '/traindev', transform=train_transform)
#         self.testImageFolder = ImageFolder(root=base_dir + '/test', transform=test_transform)

# NOTE: Cutom dataset required due to masks

class Spoof_Dataset(Dataset):
    '''
        Returns a dict with keys: 'image', 'mask', 'class_label' and values: torch.Tensor
    '''
    def __init__(self, base_dir: str, 
                 train_img_transform: T.Compose = T.Compose([T.Resize((224,224), antialias=True), T.ToTensor()]), 
                 train_mask_transform: T.Compose = T.Compose([T.Resize((224,224)), T.ToTensor()]),
                 test_img_transform: T.Compose = T.Compose([T.Resize((224,224), antialias=True), T.ToTensor()]),
                 test_mask_transform: T.Compose = T.Compose([T.Resize((224,224)), T.ToTensor()]),
                 isTrain: bool = True):
        
        super().__init__()
        self.base_dir = base_dir    

        self.mode = 'traindev' if isTrain else 'test'

        self.tr_img_tfm = train_img_transform
        self.tr_mask_tfm = train_mask_transform
        self.test_img_tfm = test_img_transform
        self.test_mask_tfm = test_mask_transform

        self.authentic_dir = os.path.join(self.base_dir, self.mode, 'authentic')

        self.copy_moved_imgs_dir = os.path.join(self.base_dir, self.mode, 'copy-moved', 'images')
        self.copy_moved_masks_dir = os.path.join(self.base_dir, self.mode, 'copy-moved', 'masks')

        self.spliced_imgs_dir = os.path.join(self.base_dir, self.mode, 'spliced', 'images')
        self.spliced_imgs_dir = os.path.join(self.base_dir, self.mode, 'spliced', 'masks')

        self.all_image_paths = os.listdir(self.authentic_dir) + os.listdir(self.copy_moved_imgs_dir) + \
            os.listdir(self.spliced_imgs_dir)
        
        self.all_masks_path = os.listdir(self.copy_moved_masks_dir) + os.listdir(self.spliced_masks_dir)


    def __len__(self):
        return len(self.all_image_paths)


    def __getitem__(self, idx: int):
        image_name = self.all_image_paths[idx]
        if image_name[0] == 'a':
            img = cv.imread(os.path.join(self.authentic_dir, image_name))
            mask = np.zeros_like(img)
            class_label = 0

        elif image_name[0] == 'c':
            img = cv.imread(os.path.join(self.copy_moved_imgs_dir, image_name))
            mask = cv.imread(os.path.join(self.copy_moved_masks_dir, image_name))
            class_label = 1
            
        elif image_name[0] == 's':
            img = cv.imread(os.path.join(self.spliced_imgs_dir, image_name))
            mask = cv.imread(os.path.join(self.spliced_masks_dir, image_name))
            class_label = 2

        # Apply custom transforms (bare minimum: Resize to 224x224 + ToTensor)
        if self.mode == 'traindev':
            img = self.tr_img_tfm(img)
            mask = self.tr_mask_tfm(mask)
        else:
            img = self.test_img_tfm(img)
            mask = self.test_mask_tfm(mask)
            
        return {'image': img, 'mask': mask, 'class_label': class_label}
