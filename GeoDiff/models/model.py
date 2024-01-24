import torch
from torch import nn

from encoders import *
from common import *

class DualEncoderEpsNetwork(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.edge_encoder_global = get_edge_encoder(config)
        self.edge_encoder_local = get_edge_encoder(config)
        
        """
        The graph neural network that extracts node-wise features.
        """
        self.encoder_global = SchNetEncoder(
            hidden_channels=config.hidden_dim,
        )