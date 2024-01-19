import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLayerPerceptron(nn.Module):
    """
    Multi-layer Perceptron.
    Note there is no activation or dropout in the last layer.
    Parameters:
        input_dim (int): input dimension
        hidden_dim (list of int): hidden dimensions
        activation (str or function, optional): activation function
        dropout (float, optional): dropout rate
    """
    
    def __init__(
        self,
        input_dim,
        hidden_dims,
        activation="relu",
        dropout=0
    ):
        super(MultiLayerPerceptron, self).__init__()
        
        self.dims = [input_dim] + hidden_dims
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        
        self.layers = nn.ModuleList()
        for i in range(len(self.dim) - 1):
            self.layers.append(nn.Linear(self.dim[i], self.dim[i + 1]))
    
    def forward(self, input):
        x = input
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1: # not final layer
                if self.activation:
                    x = self.activation(x)
                if self.dropout:
                    x = self.dropout(x)
        return x