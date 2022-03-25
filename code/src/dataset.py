import torch
import torchvision.transforms as T
from torchvision.io import read_image

import glob
import os

from .config import *


class GCD:
    # TODO: Implementar distintos tipos de augmentation para una misma imagen y hacer el return dinamico segun num_augmentations
    
    def __init__(self, data_dir, dataset_type, resize=None, aug_types=None):

        self.image_paths  = self._get_paths(data_dir, dataset_type)
        self.targets = self._get_targets()
        
        self.resize = resize
        self.aug_types = [aug_types] if isinstance(aug_types, str) else aug_types
        
        self.norm_mean = 155.5673
        self.norm_std = 70.5983

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = read_image(self.image_paths[item])
        image = (image-self.norm_mean)/self.norm_std
        
        targets = self.targets[item]
        #substract 1 since list of targets start from 1
        targets = torch.tensor(targets, dtype=torch.long) - 1

        if self.resize is not None:
            image = T.Resize(self.resize)(image)
            
        if self.aug_types is not None:
            aug_image = self._augmentate(image)
            return {
                "images": image,
                "targets": targets,
                "augmented": aug_image
            }
            

        return {
            "images": image,
            "targets": targets,
        }
    
    def _augmentate(self, img):
        
        t_dict = {
            'h_flip': T.RandomHorizontalFlip(p=0.5),
            'r_crop': T.RandomResizedCrop(size=(self.resize, 
                                                self.resize), 
                                                scale=(0.3, 0.65))
        }
        
        transform = T.Compose([v for k,v in t_dict.items() if k in self.aug_types])
        
        return transform(img)
    
    def _get_paths(self, data_dir, dataset_type):
        return glob.glob(
                         os.path.join(data_dir,f'GCD/{dataset_type}/**/*.jpg'), 
                         recursive=True
                        )
    
    def _get_targets(self):
        return list(map(int,list(map(int,[os.path.basename(x).split('_')[0] 
                                          for x in self.image_paths]))))
        
