import matplotlib
import torch.utils.data
import tqdm
import torch
import torchvision

device = "cuda:0" if torch.cuda.is_available() else "gpu"

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor],torchvision.transforms.Normalize(mean=[0.5],std=[0.5]))

path = './data'