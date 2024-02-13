import torch
import torch_geometric
from torch import nn

from ..common import MultiLayerPerceptron

class GaussianSmearingEdgeEncoder(nn.Module):
    pass

class MLPEdgeEncoder(nn.Module):
    def __init__(self, hidden_dim=100, mlp_act='relu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bond_emb = nn.Embedding(100, embedding_dim=self.hidden_dim)
        self.mlp = MultiLayerPerceptron(
            1,
            [self.hidden_dim, self.hidden_dim],
            activation=mlp_act,
        )
    
    @property
    def out_channels(self):
        return self.hidden_dim

    def forward(self, edge_length, edge_type):
        """
        Input:
            edge_length: The length of edges, shape=(E, 1).
            edge_type: The type pf edges, shape=(E,)
        Returns:
            edge_attr:  The representation of edges. (E, 2 * num_gaussians)
        """
        d_emb = self.mlp(edge_length)   # (num_edge, hidden_dim)
        edge_attr = self.bond_emb(edge_type)    # (num_edge, hidden_dim)
        return d_emb * edge_attr    # (num_edge, hidden_dim)


def get_edge_encoder(cfg):
    if cfg.edge_encoder == 'mlp':
        return MLPEdgeEncoder(cfg.hidden_dim, cfg.mlp_act)
    elif cfg.edge_encoder == 'gaussian':
        return GaussianSmearingEdgeEncoder(cfg.hidden_dim // 2, cutoff=cfg.cutoff)
    else:
        raise NotImplementedError('Unknown edge encoder: %s' % cfg.edge_encoder)
