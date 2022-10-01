import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelBinarizer
from torchvision.io import read_image
from torchvision.utils import save_image
import os
import glob
import random
import pandas as pd
import cv2
from PIL import Image 
import PIL 
from sklearn.model_selection import train_test_split

# Classes to augment    
#     396 Cc -- CCSN
#     360 St
#     301 As -- CCSN
#     285 Ns -- CCSN
#     237 Cs



def main():
    n_classes = [(0, 'Ac'),(1, 'As'),(2, 'Cb'),(3, 'Ci'),(4, 'Cc'),(5, 'Cs'),(6, 'Cu'),(7, 'Ns'),(8, 'Sc'),(9, 'St')]
    classes = {'Ac':0,'As':0,'Cb':0,'Ci':0,'Cc':0,'Cs':0,'Cu':0,'Ns':0,'Sc':0,'St':0}
    aug_classes = ['Cc', 'St', 'As', 'Ns', 'Cs']
    train_ = pd.read_csv("/data/mandonaire/Cloudappreciationsociety/filtered_images_data_aug/data_train_10c.csv")
    test = pd.read_csv("/data/mandonaire/Cloudappreciationsociety/filtered_images_data_aug/data_test_10c.csv")
    # split train into train and validation
    train = pd.DataFrame(columns=['path','class','n_class'])
    validation = pd.DataFrame(columns=['path','class','n_class'])
    for n_class, class_ in n_classes:
        data_class = train_[train_['class'] == class_]
        data_train, data_validation = train_test_split(data_class, test_size=0.125, random_state=17)
        train = pd.concat([train, data_train])
        validation = pd.concat([validation, data_validation])

    # Create folders for each set and class
    
    sets = ["train", "validation", "test"]
    for i in sets:
        os.mkdir("./" + i)
        for j in n_classes:
            os.mkdir("./" + i + "/" + j[1])
            
    train_final = pd.DataFrame(columns=['path','class','n_class'])
    validation_final = pd.DataFrame(columns=['path','class','n_class'])
    test_final = pd.DataFrame(columns=['path','class','n_class'])

    # Generate final sets

    for index, row in train.iterrows():
        pth = row['path'].split('/')
        pth.insert(5, 'train')
        img = read_image(row['path'])
        pth.pop(-1)
        pth.append('{}_{}.jpg'.format(row['class'], classes[row['class']]))
        classes[row['class']]+=1
        train_final = train_final.append({'path':'/'.join(pth),'class':row['class'],'n_class':row['n_class']}, ignore_index=True)
        img = transforms.ToPILImage()(img)
        img.save('/'.join(pth))
        img = transforms.ToTensor()(img)
        transform = transforms.Compose([transforms.ToPILImage(),
                                                transforms.RandomHorizontalFlip(p=0.5),
                                                transforms.RandomResizedCrop(size=(img.shape[1], img.shape[2]), scale=(0.75, 0.95))
                                            ])
        if row['class'] in aug_classes:
            img_aug = transform(img)
            pth.pop(-1)
            pth.append('{}_{}.jpg'.format(row['class'], classes[row['class']]))
            classes[row['class']]+=1
            train_final = train_final.append({'path':'/'.join(pth),'class':row['class'],'n_class':row['n_class']}, ignore_index=True)
            img_aug.save('/'.join(pth))

    for index, row in validation.iterrows():
        pth = row['path'].split('/')
        pth.insert(5, 'validation')
        img = read_image(row['path'])
        pth.pop(-1)
        pth.append('{}_{}.jpg'.format(row['class'], classes[row['class']]))
        classes[row['class']]+=1
        validation_final = validation_final.append({'path':'/'.join(pth),'class':row['class'],'n_class':row['n_class']}, ignore_index=True)
        img = transforms.ToPILImage()(img)
        img.save('/'.join(pth))

    for index, row in test.iterrows():
        pth = row['path'].split('/')
        pth.insert(5, 'test')
        img = read_image(row['path'])
        pth.pop(-1)
        pth.append('{}_{}.jpg'.format(row['class'], classes[row['class']]))
        classes[row['class']]+=1
        test_final = test_final.append({'path':'/'.join(pth),'class':row['class'],'n_class':row['n_class']}, ignore_index=True)
        img = transforms.ToPILImage()(img)
        img.save('/'.join(pth))

    train_final.to_csv('/data/mandonaire/Cloudappreciationsociety/filtered_images_data_aug/train.csv', index=False)
    validation_final.to_csv('/data/mandonaire/Cloudappreciationsociety/filtered_images_data_aug/validation.csv', index=False) 
    test_final.to_csv('/data/mandonaire/Cloudappreciationsociety/filtered_images_data_aug/test.csv', index=False)    

if __name__ == '__main__':
    main()


# find ./ -type f -name '*.jpg' | grep train | xargs rm -f
# find ./ -type f -name '*.jpg' | grep test | xargs rm -f
# du -a | cut -d/ -f2 | sort | uniq -c | sort -nr