import torch
import torch.nn as nn
import torchvision.transforms as T
#import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataset import MnistDataset

class fashionModel(nn.Module):
    def __init__(self):
        super(fashionModel,self).__init__()
        self.fc = nn.Linear(3*40*40, 10)
        
    
    def forward(self,x):
        x = x.view(x.size(0), -1)
        x  = self.fc(x)
        x = torch.log_softmax(x,dim=1)
        print(x.size())
        print(x[0].size)
        print(torch.sum(x[0]))
        exit()




if __name__ == "__main__":
    transforms = T.Compose([
		T.ToTensor(),
		T.Normalize((0.5,),(0.5,)),
	])
    dataset = MnistDataset('data', 'train', transforms)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8 )
    model = fashionModel()
    for images, labels in dataloader:
        pred = model(images)