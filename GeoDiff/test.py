import os
import pickle

import torch
import torch_geometric
from torch import nn
from torch_geometric.datasets import QM9

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

data_path = 'data/QM9/test_data_1k.pkl'

with open(data_path, 'rb') as f:
    batch = pickle.load(f)

print(batch)