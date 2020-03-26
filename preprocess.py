import torch
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from PIL import Image


class MNIST_loader(data.Dataset):
    """This class is needed to processing batches for the dataloader."""
    def __init__(self, data, target, transform):
        self.data = data
        self.target = target
        self.transform = transform

    def __getitem__(self, index):
        """return transformed items."""
        x = self.data[index]
        y = self.target[index]
        if self.transform:
            x = Image.fromarray(x.numpy(), mode='L')
            x = self.transform(x)
        return x, y

    def __len__(self):
        """number of samples."""
        return len(self.data)


def get_mnist(args, data_dir='./data/mnist/'):
    """get dataloders"""

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    train = datasets.MNIST(root=data_dir, train=True, download=True)
    test = datasets.MNIST(root=data_dir, train=False, download=True)

    data = train.data
    labels = train.targets

    normal_data = data[labels!=args.anormal_class]
    normal_labels = labels[labels!=args.anormal_class]

    n_train = int(normal_data.shape[0]*0.8)

    x_train = normal_data[:n_train]
    y_train = normal_labels[:n_train]              
    data_train = MNIST_loader(x_train, y_train, transform)
    dataloader_train = DataLoader(data_train, batch_size=args.batch_size, 
                                  shuffle=True, num_workers=0)
    
    anormal_data = data[labels==args.anormal_class]
    anormal_labels = labels[labels==args.anormal_class]
    x_test = torch.cat((anormal_data, normal_data[n_train:]), dim=0)
    y_test = torch.cat((anormal_labels, normal_labels[n_train:]), dim=0)
    y_test = np.where(y_test==args.anormal_class, 0, 1)
    data_test = MNIST_loader(x_test, y_test, transform)
    dataloader_test = DataLoader(data_test, batch_size=args.batch_size, 
                                  shuffle=False, num_workers=0)
    return dataloader_train, dataloader_test