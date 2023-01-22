
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt

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
    
def reshape(imgdata: pd.DataFrame()):
    img_reshaped = np.reshape(imgdata.values,(28,28))

    return img_reshaped

def drawimg(img, label=None):
    if label == None:
        plt.title(f'Label: {img[1]}')
    else:
        plt.title(f'Label: {label}')
    plt.imshow(img[0], cmap='gray')

def tensor_to_pltimg(tensor_image):
    return tensor_image.permute(1,2,0).numpy()
    
def multidrawimg(imgs, labels, batch_size: int = 16):
    fig = plt.figure(figsize=[10,10])
    plt.subplots_adjust(top=0.9, wspace=0.2, hspace=0.35)
    for i,data in enumerate(imgs):
        ax = plt.subplot(4,batch_size//4,i+1)
        ax.set_title(f'Label: {labels[i]}')
        ax.imshow(tensor_to_pltimg(data), cmap='gray')