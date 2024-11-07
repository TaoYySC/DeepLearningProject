from MNIST.src.conf import path, transform, BATCH_SIZE
import torch
import torchvision

# download the dataset for train and test
TrainData = torchvision.datasets.MNIST(path, train=True, transform=transform, download=True)
TestData = torchvision.datasets.MNIST(path, train=False, transform=transform)

# create the dataloader for train data and test data
TrainDataLoader = torch.utils.data.DataLoader(dataset=TrainData, batch_size=BATCH_SIZE, shuffle=True)
TestDataLoader = torch.utils.data.DataLoader(dataset=TestData, batch_size=BATCH_SIZE)
