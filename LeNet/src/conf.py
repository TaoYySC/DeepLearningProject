import torchvision


transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(mean=[0.5], std=[0.5])])

# 定义batch_size
from torch.utils.data import DataLoader

batch_size = 256
# 设置学习率和迭代次数
lr, num_epochs = 0.9, 10

path = 'D:\TaoProject\DeepLearningProject\LeNet\data'

TrainData = torchvision.datasets.FashionMNIST(path, train=True, transform=transform,download=False)
TestData = torchvision.datasets.FashionMNIST(path, train=False, transform=transform,download=False)

train_iter = DataLoader(TrainData, batch_size=batch_size, shuffle=True)
test_iter = DataLoader(TestData, batch_size=batch_size, shuffle=False)
