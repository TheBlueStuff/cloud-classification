import cv2
import numpy as np
import glob
import os
from random import sample
import torchvision.transforms as T


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
