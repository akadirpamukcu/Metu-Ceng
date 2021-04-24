import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from dataset import MnistDataset
import numpy as np
class fashionModel(nn.Module): # 1 layer simple
    def __init__(self):
        super(fashionModel,self).__init__()
        self.fc1 = nn.Linear(3*40*40, 1024)
        self.fc2 = nn.Linear(1024, 10)
        
    
    def forward(self,x):
        x = torch.flatten(x,1)
        x  = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.log_softmax(x,dim=1)
        return x


def train(model, optimizer, train_dataloader, valid_dataloader, epochs, device,model_name):
    avg_training_losses = []
    avg_validation_losses = []
    min_loss=1000
    min_delta=0.001
    count =0
    flag =False
    for epoch_id in range(epochs):
        model.train()
        training_losses = []
        validation_losses = []
        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            pred = model(images)
            loss = F.nll_loss(pred,labels)
            loss.backward()
            optimizer.step()
            print(loss.item())
            training_losses.append(loss.item())
        with torch.no_grad():
            model.eval()
            correct_num=0
            total_num=0            
            for images, labels in valid_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                pred = model(images)
                loss = F.nll_loss(pred,labels)
                loss.backward()
                _, predicted_labes = torch.max(pred, 1)
                correct_num += (predicted_labes == labels).sum()
                total_num=labels.size(0)
                validation_losses.append(loss.item())
            
        validation_loss = np.average(validation_losses)
        avg_validation_losses.append(validation_loss)
        avg_training_losses.append(np.average(training_losses))

        if(validation_loss > min_loss):
            count+=1
            flag = true if (count > 5) else false
            if(flag):
                print( "EARLY STOP YEAH")
                break

        elif(validation_loss < (min_loss - min_delta)):
            min_loss = validation_loss
            count=0

    torch.save(model.state_dict(), model_name + "_acc:" + str((100 * correct_num) / (total_num + 1)) )    
    print("saved model name is:    ", model_name + "_acc:" + str((100 * correct_num) / (total_num + 1)))
    print("AVG LOSSES ARE:::\n")
    print("avg training losess:")
    for i in avg_training_losses:
        print(i)
    print("avg validation losess:")
    for i in avg_validation_losses:
        print(i)







        


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(31)
    transforms = T.Compose([
		T.ToTensor(),
		T.Normalize((0.5,),(0.5,)),
	])
    dataset = MnistDataset('data', 'train', transforms)
    train_dataset, valid_dataset = random_split(dataset, [int(len(dataset)*.8), int(len(dataset)*.2)])
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8 )
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True, num_workers=8 )
    model = fashionModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    train(model, optimizer, train_dataloader, valid_dataloader, 10, device, "model1")
    
