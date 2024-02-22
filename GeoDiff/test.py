import os
import pickle
import yaml
from easydict import EasyDict

import torch
import torch_geometric

from models.espnet import get_model
from models.encoder import CFConv

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'[device]: {device}')

data_path = '../../Data/temp/batch.pkl'

with open(data_path, 'rb') as f:
    batch = pickle.load(f)

fig_path = 'configs/qm9_default.yml'

with open(fig_path, 'r') as f:
    config = EasyDict(yaml.safe_load(f))

model = get_model(config.model)

model(
    atom_type=batch.atom_type,
    pos = batch.pos,
    bond_index = batch.edge_index,
    bond_type = batch.edge_type,
    batch = batch.batch,
    time_step = 0,
)

print(model.betas.shape, model.alphas.shape)
print(model.betas)
print(model.alphas)