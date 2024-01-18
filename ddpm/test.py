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

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print("cuda: ", torch.cuda.is_available())

model = Unet(img_ch).to(device)

print("Num params: ", sum(p.numel() for p in model.parameters()))