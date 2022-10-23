import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from metadata import *

class CNN():
    def __init__(self):
        device = torch.device('cpu')
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        in_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(in_ftrs, len(classes))
        in_ftrs = self.model.AuxLogits.fc.in_features
        self.model.AuxLogits.fc = nn.Linear(in_ftrs, len(classes))
        self.model.load_state_dict(torch.load("./pretrained/model.pth", map_location=device))
        self.model.eval()
        input_size = 299
        self.transform = A.Compose(
            [
                A.Resize (height=input_size, width=input_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )