import matplotlib
import torch.utils.data
import tqdm
import torch
import torchvision

# data path
path = './data'

# define batch_size as 256
BATCH_SIZE = 64

# EPOCH
EPOCHS = 10

# check gpu
device = "cuda:0" if torch.cuda.is_available() else "cpu"

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(mean=[0.5], std=[0.5])])