import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from dataset import MnistDataset
import numpy as np
from models import *
import os
from PIL import Image

class TestDataset(Dataset):
    def __init__(self, dataset_path, split,transforms):
        data_path = os.path.join(dataset_path, split)
        
        self.data = []
        with open(os.path.join(data_path, 'labels.txt'), 'r') as f:
            for line in f:
                img_name = line.split("\n")[0]
                img_path = os.path.join(data_path, img_name)
                self.data.append((img_path))
        self.transforms = transforms


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path = self.data[index]
        img = Image.open(img_path)
        img = self.transforms(img)
        return img
if __name__ == "__main__":
    transforms = T.Compose([
		T.ToTensor(),
		T.Normalize((0.5,),(0.5,)),
	])
    device = torch.device('cuda')
    ls = []
    torch.manual_seed(42)
    dataset = TestDataset('data', 'test', transforms)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
    model = ThreeLayerRelu(1024)
    model.to(device)
    model.load_state_dict(torch.load( "/home/kadir/Documents/Metu-Ceng/ceng499_hw1/bestacc:55.0"   ))
    model.eval()
    lines=[]

    data_path = 'data'
    with torch.no_grad():
        for image in dataloader:
            image = image.to(device)
            pred = model(image)
            images, labels = torch.max(pred,1)
            for i in range(len(image)):
                ls.append(int(labels[i]))
    
    with open("data/labels.txt", 'r') as fx:
            for line in fx:
                lines.append(line.split("\n")[0])
    for j in range(len(dataset)):
        print(lines[j]+ " " +str(ls[j]))
