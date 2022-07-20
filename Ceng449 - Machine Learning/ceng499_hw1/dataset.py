import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

class MnistDataset(Dataset):
    def __init__(self, dataset_path, split,transforms):
        data_path = os.path.join(dataset_path, split)
        self.data = []
        with open(os.path.join(data_path, 'labels.txt'), 'r') as f:
            for line in f:
                img_name, label = line.split()
                img_path = os.path.join(data_path, img_name)
                label = int(label)
                self.data.append((img_path, label))
        self.transforms = transforms


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path = self.data[index][0]
        label = self.data[index][1]
        img = Image.open(img_path)
        img = self.transforms(img)
        return img,label

if __name__ == "__main__":
    transforms = T.Compose([
		T.ToTensor(),
		T.Normalize((0.5,),(0.5,)),
	])

    dataset = MnistDataset('data', 'train', transforms)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8 )
    print(len(dataset))
    print(dataset[0][0].size())
    for i, l in dataloader:
        print(i.size())
        print(l)
        break
    #print(len(dataset))
    #print(dataset[0])

        
    
