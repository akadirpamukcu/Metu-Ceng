import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataset import MnistDataset

class OneLayer(nn.Module):
    def __init__(self):
        super(OneLayer,self).__init__()
        self.fc = nn.Linear(3*40*40, 10)
        
    
    def forward(self,x):
        x = x.view(x.size(0), -1)
        x  = self.fc(x)
        x = torch.log_softmax(x,dim=1)
        return x

class TwoLayerRelu(nn.Module): # 1 layer simple
    def __init__(self,hidden_size):
        super(TwoLayerRelu,self).__init__()
        self.fc1 = nn.Linear(3*40*40, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)
        
    
    def forward(self,x):
        x = torch.flatten(x,1)
        x  = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.log_softmax(x,dim=1)
        return x

class TwoLayerTanh(nn.Module): # 1 layer simple
    def __init__(self,hidden_size):
        super(TwoLayerTanh,self).__init__()
        self.fc1 = nn.Linear(3*40*40, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)
        
    
    def forward(self,x):
        x = torch.flatten(x,1)
        x  = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        x = torch.log_softmax(x,dim=1)
        return x

class TwoLayerSigmoid(nn.Module): # 1 layer simple
    def __init__(self,hidden_size):
        super(TwoLayerSigmoid,self).__init__()
        self.fc1 = nn.Linear(3*40*40, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)
        
    
    def forward(self,x):
        x = torch.flatten(x,1)
        x  = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        x = torch.log_softmax(x,dim=1)
        return x

class ThreeLayerRelu(nn.Module): # 1 layer simple
    def __init__(self,hidden_size):
        super(ThreeLayerRelu,self).__init__()
        self.fc1 = nn.Linear(3*40*40, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 10)
    
    def forward(self,x):
        x = torch.flatten(x,1)
        x  = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = torch.log_softmax(x,dim=1)
        return x

class ThreeLayerTanh(nn.Module): # 1 layer simple
    def __init__(self,hidden_size):
        super(ThreeLayerTanh,self).__init__()
        self.fc1 = nn.Linear(3*40*40, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 10)
    
    def forward(self,x):
        x = torch.flatten(x,1)
        x  = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        x = F.tanh(x)
        x = self.fc3(x)
        x = torch.log_softmax(x,dim=1)
        return x

class ThreeLayerSigmoid(nn.Module): # 1 layer simple
    def __init__(self,hidden_size):
        super(ThreeLayerSigmoid,self).__init__()
        self.fc1 = nn.Linear(3*40*40, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 10)
    
    def forward(self,x):
        x = torch.flatten(x,1)
        x  = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        x = self.fc3(x)
        x = torch.log_softmax(x,dim=1)
        return x



if __name__ == "__main__":
    transforms = T.Compose([
		T.ToTensor(),
		T.Normalize((0.5,),(0.5,)),
	])
    dataset = MnistDataset('data', 'train', transforms)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4 )
