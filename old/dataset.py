import albumentations
import torch
import numpy as np

import cv2
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageClassificationDataset:
    def __init__(self, image_paths, targets, normalization, resize=None):

        self.image_paths = image_paths
        self.targets = targets
        self.normalization = normalization
        self.resize = resize

        self.aug = albumentations.Compose([albumentations.Normalize(always_apply=True)])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = cv2.imread(self.image_paths[item])
        targets = self.targets[item]

        if self.resize is not None:
            image = cv2.resize(image, (self.resize[1], self.resize[0]))

        image = np.array(image)
        image = (image - self.normalization[0]) / self.normalization[1]

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            "images": torch.tensor(image, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.long),
        }
