import math
import cv2
import numpy as np
import glob
import os
from random import sample
import torch
import torch.nn as nn
import torchvision.transforms as T

from . import config
from . import dataset

from .models import graph_nets


########### FUNCTIONS ###########

def readImage(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (256, 256))
    image = np.array(image)
    image = np.transpose(image, (2, 0, 1)).astype(np.float32) #C,H,W
    return image

def get_gcd_paths(data_dir, dataset_type):
    return glob.glob(
                     os.path.join(data_dir,f'GCD/{dataset_type}/**/*.jpg'), 
                     recursive=True
                    )

def get_grscd_paths(data_dir, dataset_type):
    return glob.glob(
                     os.path.join(data_dir,f'GRSCD/{dataset_type}/**/*.jpg'), 
                     recursive=True
                    )

def get_ccsn_paths(data_dir):
    return glob.glob(
                     os.path.join(data_dir,f'CCSN_v2/**/*.jpg'), 
                     recursive=True
                    )

def get_targets_ccsn(image_paths):
    return list(list([os.path.basename(x).split('-')[0] for x in image_paths]))

def random_sample(data, fraction=0.5):
    num_samples = int(fraction*len(data))
    
    return sample(data, num_samples)

def augmentate(img, aug_types ,resize=256):
    t_dict = {
            'h_flip': T.RandomHorizontalFlip(p=1),
            'v_flip': T.RandomHorizontalFlip(p=1),
            'rot': T.RandomRotation(15),
            'r_crop': T.RandomResizedCrop(size=(resize, 
                                                resize), 
                                                scale=(0.3, 0.65))
        }
        
    transform = T.Compose([v for k,v in t_dict.items() if k in aug_types])
    return transform(img)


########### LOSS ###########

def loge_loss(x , labels):
    epsilon = 1 - math.log(2)
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    loss = criterion(x, labels)
   
    loss = torch.mean(torch.log(epsilon + loss) - math.log(epsilon))
    return loss

########### SWEEPS ###########

def build_dataset_gcd(batch_size):
    
    ### TRAIN
    path_train_images = get_gcd_paths(config.DATA_DIR,'train')
    
    train_dataset = dataset.GCD(path_train_images, resize=256)

    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=4,
            shuffle=True,
        )
    
    ### TEST
    path_test_images = get_gcd_paths(config.DATA_DIR,'test')

    test_dataset = dataset.GCD(path_test_images, resize=256)

    test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=4,
            shuffle=False,
        )
    
    ### AUGMENTATION
    augmentation_datasets = [
        dataset.GCD(
                    random_sample(path_train_images, fraction=1), 
                    resize=256, 
                    aug_types=atype)
        for atype in config.AUGMENTATION_TYPES
    ]
    
    augmentation_loaders = [
        torch.utils.data.DataLoader(
                aug_dataset,
                batch_size=batch_size,
                num_workers=4,
                shuffle=True,
            )
    for aug_dataset in augmentation_datasets
    ]
    
    return train_loader, test_loader, augmentation_loaders

def build_model_gatconv(num_classes, hid_dim, num_hidden, num_heads, threshold, device):
    return graph_nets.GATConvGNN(num_classes, hid_dim, num_hidden, num_heads, threshold).to(device)

def build_optimizer(optim_type, model, lr):
    
    if optim_type=='adam':
        optim=torch.optim.Adam(model.parameters(), lr)
        
    elif optim_type=='sgd':
        optim=torch.optim.SGD(model.parameters(), lr, momentum=0.9)
        
    elif optim_type=='nadam':
        optim=torch.optim.NAdam(model.parameters(), lr)
        
    else:
        raise NotImplementedError(f"{optim_type} is not a valid optimizer")
    
    return optim


def build_criterion(criterion_type):
    
    if criterion_type=='cross_entropy':
        criterion=nn.CrossEntropyLoss()
        
    elif criterion_type=='loge':
        criterion=loge_loss
        
    else:
        raise NotImplementedError(f"{criterion_type} is not a valid criterion")
    
    return criterion