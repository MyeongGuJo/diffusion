import torch
import torch_geometric
from torch import nn

class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()
        
    def forward(self, x):
        return nn.functional.softplus(x) - self.shift

class CFConv(torch_geometric.nn.MessagePassing):
    def __init__(self, in_channels, out_channels, num_filters,
                 nn, cutoff, smooth):
        super(CFConv, self).__init__(aggr='add')
        self.lin1 = nn.Linear(in_channels, num_filters, bias=False)
        self.lin2 = nn.Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff
        self.smooth = smooth
        
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
    
    def forward(self, x, edge_index, edge_length, edge_attr):
        if self.smooth:
            pass
    
    def message(self, x_j, W):
        return x_j * W

class InteractionBlock(nn.Module):
    def __init__(self, hidden_channels, num_gaussians,
                 num_filters, cutoff, smooth):
        super(InteractionBlock, self).__init__()
        mlp = nn.Sequential(
            nn.Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            nn.Linear(num_filters, num_filters)
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters,
                           mlp, cutoff, smooth)

class SchNetEncoder(nn.Module):
    
    def __init__(self, hidden_channels=128, num_filters=128,
                 num_interactions=6, edge_channels=100, cutoff=10.0,
                 smooth=False):
        super().__init__()
        
        self. hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.cutoff = cutoff
        
        # max_norm = 10.0?
        self.embedding = nn.Embedding(100, hidden_channels,
                                      max_norm=10.0)
        
        self.num_interactions = nn.ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, edge_channels,
                                     num_filters, cutoff, smooth)
            self.interactions.append(block)