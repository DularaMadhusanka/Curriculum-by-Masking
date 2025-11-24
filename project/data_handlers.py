import cv2
import torch
import torchvision
from torch import randperm, default_generator
from torch.utils.data import Subset, DataLoader, Dataset
from torchvision.transforms import transforms
from einops import rearrange
import numpy as np


class CIFAR(torch.utils.data.Dataset):
    def __init__(self, dataset_path='./data', transformations=None, should_download=True,data_set=torchvision.datasets.Food101):
        self.dataset_train = data_set(dataset_path, download=should_download,split = "train")
        self.transformations = transformations
        self.probs=[]
        for i in range(len(self.dataset_train)):
            (imeg, laebl) = self.dataset_train[i]
            self.probs.append(self.get_probability(imeg))
    def get_probability(self, img):
        bla=cv2.CV_64F
        
        transformz = transforms.Compose([
        transforms.Resize((224,224))] #64 for non cvt
        )
        img = transformz(img)
        innp=np.asarray(img)

        dx=cv2.Sobel(innp,bla,1,0)
        dy=cv2.Sobel(innp,bla,0,1)
        mag=np.sqrt(dx**2+dy**2)
        mag=torch.Tensor(mag.swapaxes(0,-1).swapaxes(1,2))
        p=4
        image2=rearrange(mag, 'c (p1 w) (p2 h) -> (p1 p2) w h c', p1=p, p2=p)
        image2=image2.numpy()
        tempor=[]
        for kk in range(len(image2)):
            tempor.append(np.mean(image2[kk]))
        tempor=np.asarray(tempor)
        tempor=tempor/sum(tempor)
        return tempor
    def __getitem__(self, index):
        (img, label) = self.dataset_train[index]
        if self.transformations is not None:
            return self.transformations(img), label, self.probs[index]
        return img, label
    def __len__(self):
        return len(self.dataset_train)

def get_train_val_loaders_food101(batch_size=64, dataset=torchvision.datasets.Food101,resize=32):
    transform = transforms.Compose([
        transforms.Resize((resize,resize)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )
    train_data=Food101(transformations=transform, data_set=dataset)
    #train_ds = Subset(train_data, indices[0: len(train_data)])
    print(f'Train data size {len(train_data)}')
    train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return train_loader

def get_test_loader_food101(batch_size=64, dataset=torchvision.datasets.CIFAR10,resize=32):
    test_changes = transforms.Compose([transforms.Resize((resize,resize)),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))])
    test_data = dataset(root='./data', split='test', download=True, transform=test_changes)
    test_loader = DataLoader(test_data, batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return test_loader

class CIFAR(torch.utils.data.Dataset):
    """CIFAR-10 dataset with probability computation for curriculum masking."""
    
    def __init__(self, dataset_path='./data', transformations=None, should_download=True, data_set=torchvision.datasets.CIFAR10):
        self.dataset_train = data_set(dataset_path, download=should_download)
        self.transformations = transformations
        self.probs = []
        for i in range(len(self.dataset_train)):
            (img, label) = self.dataset_train[i]
            self.probs.append(self.get_probability(img))
    
    def get_probability(self, img):
        """Compute patch importance scores based on image gradients."""
        bla = cv2.CV_64F
        transformz = transforms.Compose([transforms.Resize(32)])
        img = transformz(img)
        innp = np.asarray(img)
        
        dx = cv2.Sobel(innp, bla, 1, 0)
        dy = cv2.Sobel(innp, bla, 0, 1)
        mag = np.sqrt(dx**2 + dy**2)
        mag = torch.Tensor(mag.swapaxes(0, -1).swapaxes(1, 2))
        
        # Divide into 4x4 patches
        image2 = rearrange(mag, 'c (p1 w) (p2 h) -> (p1 p2) w h c', p1=4, p2=4)
        image2 = image2.numpy()
        
        # Compute mean gradient per patch
        tempor = []
        for kk in range(len(image2)):
            tempor.append(np.mean(image2[kk]))
        tempor = np.asarray(tempor)
        tempor = tempor / sum(tempor)
        return tempor
    
    def __getitem__(self, index):
        (img, label) = self.dataset_train[index]
        if self.transformations is not None:
            return self.transformations(img), label, self.probs[index]
        return img, label
    
    def __len__(self):
        return len(self.dataset_train)
def get_train_val_loaders_cifar(val_size=5000, batch_size=64, dataset=torchvision.datasets.CIFAR10, resize=32):
    """Get train and validation dataloaders for CIFAR-10."""
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(resize),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_data = CIFAR(transformations=transform, data_set=dataset)
    val_data = dataset(root='./data', train=True, download=True, transform=val_transforms)
    
    torch.manual_seed(33)
    train_size = len(train_data) - val_size
    indices = randperm(sum([train_size, val_size]), generator=default_generator).tolist()
    train_ds = Subset(train_data, indices[0: train_size])
    val_ds = Subset(val_data, indices[train_size: train_size + val_size])
    
    print(f'Train data size {len(train_ds)}, Validation data size {len(val_ds)}')
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return train_loader, val_loader


def get_test_loader_cifar(batch_size=64, dataset=torchvision.datasets.CIFAR10, resize=32):
    """Get test dataloader for CIFAR-10."""
    test_changes = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_data = dataset(root='./data', train=False, download=True, transform=test_changes)
    test_loader = DataLoader(test_data, batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return test_loader