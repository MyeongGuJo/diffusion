import os
import pickle
import torch
from torch import nn

print(os.getcwd())

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print(device)

with open('../../geodiff/temp/batch.pkl', 'rb') as f:
    batch = pickle.load(f)

print(batch)