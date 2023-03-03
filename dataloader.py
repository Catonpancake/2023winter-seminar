import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class DigitDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.data = pd.read_csv(img_dir)
        self.istrain = "label" in self.data.columns
        self.img_dir = img_dir
        self.img = self.data if not self.istrain else self.data.drop("label", axis=1)
        self.img = [np.reshape(row[1].values,(28,28)) for row in self.img.iterrows()]
        self.transform = transform
        
    def __len__(self):
        return len(self.img)

    def __getitem__(self,idx: int):
        image = self.img[idx]
        if self.transform:
            image = self.transform(image.astype(float))
        if self.istrain:
            label = self.data['label'][idx]
            
            return image, label
        else:
            return image
        
        
