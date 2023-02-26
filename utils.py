import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import timm
import csv

import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt

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
    
def multidrawimg(imgs, labels, batch_size: int=16):
    fig = plt.figure(figsize=[10,10])
    plt.subplots_adjust(top=0.9, wspace=0.2, hspace=0.35)
    for i,data in enumerate(imgs):
        ax = plt.subplot(4,batch_size//4,i+1)
        ax.set_title(f'Label: {labels[i]}')
        ax.imshow(tensor_to_pltimg(data), cmap='gray')
        
        
def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device).float(), y.to(device)

        # 예측 오류 계산
        pred = model(X)
        loss = loss_fn(pred, y)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
def test(dataloader, model, loss_fn, device, valid=True):
    
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        if valid:
            for X, y in dataloader:
                X, y = X.to(device).float(), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            test_loss /= num_batches
            correct /= size
            print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        elif valid == False:
            with open('Submission.csv', 'a+', newline='') as f:  # 'a+' means append to a file
                thewriter = csv.writer(f)
                for X in dataloader:
                    X = X.to(device).float()
                    pred = model(X)
                    result = pred.argmax().item()  # this your label
                    print(result)
                    thewriter.writerow([result])