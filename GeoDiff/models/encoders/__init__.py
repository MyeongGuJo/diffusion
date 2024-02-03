import torch
import torch_geometric
from torch import nn

from schnet import *
from edges import *

def get_edge_encoder(cfg):
    if cfg.edge_encoder == 'mlp':
        return MLPEdgeEncoder(cfg.hidden_dim, cfg.mlp_act)
    elif cfg.edge_encoder == 'gaussian':
        return GaussianSmearingEdgeEncoder(cfg.hidden_dim // 2, cutoff=cfg.cutoff)
    else:
        raise NotImplementedError('Unknown edge encoder: %s' % cfg.edge_encoder)
