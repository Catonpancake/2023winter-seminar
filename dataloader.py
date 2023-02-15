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
        self.img_labels = pd.read_csv(img_dir)['label']
        self.img_dir = img_dir
        self.img = [np.reshape(row[1].values[1:],(28,28)) for row in pd.read_csv(img_dir).iterrows()]
        self.transform = transform
        # self.target_transform = target_transform
        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self,idx):
        image = self.img[idx]
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image.astype(float))
        # if self.target_transform:
        #     label = self.target_transform(label)
        return image, label
        
