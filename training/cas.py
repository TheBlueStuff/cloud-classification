from torch.utils.data import Dataset
import pandas as pd
import cv2

class CAS(Dataset):
    """CAS Dataset."""
    def __init__(self, transform=None, data_path=None):

        self.classes = [
            (0, 'Ac'),
            (1, 'As'),
            (2, 'Cb'),
            (3, 'Ci'),
            (4, 'Cc'),
            (5, 'Cs'),
            (6, 'Cu'),
            (7, 'Ns'),
            (8, 'Sc'),
            (9, 'St')
        ]
        
        self.transform = transform
        self.data = pd.read_csv(data_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_row = self.data.loc[idx, :]
        try:
            image = cv2.imread(image_row['path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label = image_row['n_class']
        except:
            print(image_row['path'])
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label
    
    @staticmethod
    def get_n_classes():
        return 10