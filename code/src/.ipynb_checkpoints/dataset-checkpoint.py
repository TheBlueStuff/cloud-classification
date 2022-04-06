import torch
import torchvision.transforms as T
from torchvision.io import read_image

import glob
import os

from . import config
from . import utils


class GCD:
    def __init__(self, image_paths, resize=None, aug_types=None):

        self.image_paths  = image_paths
        self.targets = self._get_targets()
        
        self.resize = resize
        #self.aug_types = [aug_types] if isinstance(aug_types, str) else aug_types
        self.aug_types = aug_types
    
        self.norm_mean = 155.5673
        self.norm_std = 70.5983
        
        self.aug_transform = T.Compose([T.RandomHorizontalFlip(),
                      T.RandomHorizontalFlip(),
                      T.RandomRotation(15),
                      ])
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = read_image(self.image_paths[item]).float()
        image = (image-self.norm_mean)/self.norm_std
        
        # image = T.Compose([
        #             T.ToPILImage(),
        #             T.ToTensor(),
        #             T.Normalize(
        #                 mean=[0.485, 0.456, 0.406],
        #                 std=[0.229, 0.224, 0.225],
        #             )
        #             ])(image)

        targets = self.targets[item]
        #substract 1 since list of targets start from 1
        targets = torch.tensor(targets, dtype=torch.long) - 1

        if self.resize is not None:
            image = T.Resize(self.resize)(image)
            
        if self.aug_types is not None:
            #image = utils.augmentate(image, self.aug_types, self.resize)
            image = self.aug_transform(image)

        return {
            "images": image,
            "targets": targets,
        }
    
    
    
    def _get_targets(self):
        return list(map(int,list(map(int,[os.path.basename(x).split('_')[0] 
                                          for x in self.image_paths]))))
    
    

class CCSN:
    def __init__(self, image_paths, targets, resize=None, aug_types=None):

        self.image_paths  = image_paths
        self.targets = targets
        
        self.resize = resize
        self.aug_types = [aug_types] if isinstance(aug_types, str) else aug_types

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = read_image(self.image_paths[item]).float()
        image = T.Compose([
                    T.ToPILImage(),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    )
                    ])(image)

        targets = self.targets[item]

        if self.resize is not None:
            image = T.Resize(self.resize)(image)
            
        if self.aug_types is not None:
            image = utils.augmentate(image, self.aug_types, self.resize)

        return {
            "images": image,
            "targets": targets,
        }

    
class GRSCD:
    def __init__(self, image_paths, resize=None, aug_types=None):

        self.image_paths  = image_paths
        self.targets = self._get_targets()
        
        self.resize = resize
        self.aug_types = [aug_types] if isinstance(aug_types, str) else aug_types

        self.norm_mean = 155.5673
        self.norm_std = 70.5983
        
        
        self.aug_transform = T.Compose([T.RandomHorizontalFlip(),
                      T.RandomHorizontalFlip(),
                      T.RandomRotation(15),
                      ])
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = read_image(self.image_paths[item]).float()

        image = T.Compose([
                    T.ToPILImage(),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    )
                    ])(image)

        targets = self.targets[item]
        #substract 1 since list of targets start from 1
        targets = torch.tensor(targets, dtype=torch.long) - 1

        if self.resize is not None:
            image = T.Resize(self.resize)(image)
            
        if self.aug_types is not None:
            #image = utils.augmentate(image, self.aug_types, self.resize)
            image = T.Compose([T.RandomHorizontalFlip(),
                      T.RandomHorizontalFlip(),
                      T.RandomRotation(15),
                      ])(image)

        return {
            "images": image,
            "targets": targets,
        }
    
    
    
    def _get_targets(self):
        return list(map(int,list(map(int,[os.path.basename(x).split('_')[0] 
                                          for x in self.image_paths]))))