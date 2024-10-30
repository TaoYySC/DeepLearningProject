import pandas as pd
import numpy as np
import sklearn


TrainData = torchvision.datasets.MNIST(path,train=True,transform = transform,download=True)
TestData = torchvision.datasets.MNIST(path,train=False,transform = transform)

BATCH_SIZE = 256

TrainDataLoader = torch.utils.data.DataLoader(dataset=TrainData,batch_size=BATCH_SIZE,shuffle=True)
TestDataLoader = torch.utils.data.DataLoader(dataset=TestData,batch_size=BATCH_SIZE)



if __name__ == '__main__':
    print("my demo")