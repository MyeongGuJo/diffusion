import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

# Visualization tools
import graphviz
from torchview import draw_graph
import torchvision
from torchvision import transfroms
import matplotlib.pyplot as plt

from model.model import Unet
from utils.utils import show_images

NUM_CLASSES = 10
IMG_SIZE = 16
IMG_CH = 1
BATCH_SIZE = 128

train_set = torchvision.dataset.FashionMNIST(
    "./data/", download=True, transform=transforms.Compose([transforms.ToTensor()])
)

show_images(train_set)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print("cuda: ", torch.cuda.is_available())

model = Unet(img_ch).to(device)

print("Num params: ", sum(p.numel() for p in model.parameters()))