import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from dataset import MnistDataset
from models import *
import numpy as np

best_accuracy = 0
best_name= ""
best_list = []

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
    global best_accuracy
    global best_name
    global best_list
    avg_training_losses = []
    avg_validation_losses = []
    min_loss=1000
    min_delta=0.001
    count =0
    flag =False
    for epoch_id in range(epochs):
        print("EPOCH", epoch_id)
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
            training_losses.append(loss.item())
        with torch.no_grad():
            model.eval()
            correct_num=0
            total_num=0            
            for images, labels in valid_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                pred = model(images)
                v_loss = F.nll_loss(pred,labels)
                _, predicted_labes = torch.max(pred, 1)
                correct_num += (predicted_labes == labels).sum()
                total_num+=labels.size(0)
                validation_losses.append(v_loss.item())
            
        validation_loss = np.average(validation_losses)
        avg_validation_losses.append(validation_loss)
        avg_training_losses.append(np.average(training_losses))

        if(validation_loss > min_loss):
            count+=1
            flag = True if (count > 5) else False
            if(flag):
                print("EARLY STOP")
                break

        elif(validation_loss < (min_loss - min_delta)):
            min_loss = validation_loss
            count=0

    torch.save(model.state_dict(), model_name + "acc:" + str(float((100 * correct_num) / (total_num + 1))) )    
    print("saved model name is: ", model_name + "acc:" + str( float((100 * correct_num) / (total_num + 1)) ))
    accuracy = (100 * correct_num) / (total_num + 1)
    if (accuracy>best_accuracy):
        best_accuracy = accuracy
        best_name = model_name
        best_list = [avg_training_losses,avg_validation_losses]
    best_accuracy = accuracy if (accuracy>best_accuracy) else accuracy
    print('Percent correct: %.5f %%' % ((100 * correct_num) / (total_num + 1)))
    print("AVG LOSSES ARE:::")
    print("avg training losess:")
    for i in avg_training_losses:
        print(i)
    print("avg validation losess:")
    for i in avg_validation_losses:
        print(i)
        


if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    torch.manual_seed(42)
    transforms = T.Compose([
		T.ToTensor(),
		T.Normalize((0.5,),(0.5,)),
	])
    learning_rates = [0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003]
    hidden_layer_sizes = [256,512,1024]
    dataset = MnistDataset('data', 'train', transforms)
    train_dataset, valid_dataset = random_split(dataset, [int(len(dataset)*.8), int(len(dataset)*.2)])
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4 )
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True, num_workers=4 )
    epoch = 10
    for l_r in learning_rates:
        #1 layer model
        print("\nOne layer model is training..\n"+ "lr: " + str(l_r))
        model = OneLayer()
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=l_r)
        model_name = "1_layer_lr: "+ str(l_r) + "_"
        train(model, optimizer, train_dataloader, valid_dataloader, epoch, device, model_name)
        print("\nOne layer model is done..\n")


    for lr in learning_rates:
        for l_size in hidden_layer_sizes:
            #2 layers
            #   RELU
            print("\n2_layer_RELU model is training..\n"+ "lr: " , lr ,  " h_size:", l_size )
            model = TwoLayerRelu(l_size)
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            model_name = "2_layer_RELU_lr: "+ str(lr) + "_" + "h_size: " + str(l_size) 
            train(model, optimizer, train_dataloader, valid_dataloader, epoch, device, model_name)
            print("\n2_layer_RELU model is done..\n"+ "lr: " , lr , " h_size:", l_size )
            #  Tanh
            print("\n2_layer_tanh model is training..\n"+ "lr: "  , lr , " h_size:", l_size )
            model = TwoLayerTanh(l_size)
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            model_name = "2_layer_Tanh_lr: "+ str(lr) + "_" + "h_size: " + str(l_size) 
            train(model, optimizer, train_dataloader, valid_dataloader, epoch, device, model_name)
            print("\n2_layer_Tanh model is done..\n")
            #  Sigmoid
            print("\n2_layer_Sigmoid model is training..\n"+ "lr: "  , lr , " h_size:", l_size )
            model = TwoLayerTanh(l_size)
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            model_name = "2_layer_Sigmoid_lr: "+ str(lr) + "_" + "h_size: " + str(l_size) 
            train(model, optimizer, train_dataloader, valid_dataloader, epoch, device, model_name)
            print("\n2_layer_Sigmoid model is done..\n")
            #3 layers
            #   RELU
            print("\n3_layer_RELU model is training..\n"+ "lr: "  , lr , " h_size:", l_size )
            model = ThreeLayerRelu(l_size)
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            model_name = "3_layer_RELU_lr: "+ str(lr) + "_" + "h_size: " + str(l_size) 
            train(model, optimizer, train_dataloader, valid_dataloader, epoch, device, model_name)
            print("\n3_layer_RELU model is done..\n")
            #  Tanh
            print("\n3_layer_tanh model is training..\n" + "lr: "  , lr ," h_size:", l_size )
            model = ThreeLayerTanh(l_size)
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            model_name = "3_layer_Tanh_lr: "+ str(lr) + "_" + "h_size: " + str(l_size) 
            train(model, optimizer, train_dataloader, valid_dataloader, epoch, device, model_name)
            print("\n3_layer_Tanh model is done..\n")
            #  Sigmoid
            print("\n3_layer_Sigmoid model is training..\n"+ "lr: "  , lr ," h_size:", l_size )
            model = ThreeLayerTanh(l_size)
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            model_name = "3_layer_Sigmoid_lr: "+ str(lr) + "_" + "h_size: " + str(l_size) 
            train(model, optimizer, train_dataloader, valid_dataloader, epoch, device, model_name)
            print("\n3_layer_Sigmoid model is done..\n"+ "lr: "  , lr , " h_size:", l_size )
    print("best_name: ", best_name, "its accuracy: " best_accuracy, "\n","\n" best_list)

    
