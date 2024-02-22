import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import radius_graph, radius
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_sparse import coalesce

from utils.chem import BOND_TYPES

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
    
    def __init__(self, input_dim, hidden_dims, activation='relu', dropout=0):
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
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
        
    def forward(self, input):
        x = input
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
            if self.dropout:
                x = self.dropout(x)
        return x
    
def assemble_atom_pair_feature(node_attr, edge_index, edge_attr):
    h_row, h_col = node_attr[edge_index[0]], node_attr[edge_index[1]]
    h_pair = torch.cat([h_row*h_col, edge_attr], dim=-1)    # (E, 2H)
    return h_pair

def _extend_graph_order(
    num_nodes, edge_index, edge_type, order=3
):
    """
    Args:
        num_nodes:  Number of atoms.
        edge_index: Bond indices of the original graph.
        edge_type:  Bond types of the original graph.
        order:  Extension order.
    Returns:
        new_edge_index: Extended edge indices.
        new_edge_type:  Extended edge types.
    """
    def binarize(x):
        return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

    def get_higher_order_adj_matrix(adj, order):
        """
        Args:
            adj:        (N, N)
            type_mat:   (N, N)
        Returns:
            Following attributes will be updated:
              - edge_index
              - edge_type
            Following attributes will be added to the data object:
              - bond_edge_index:  Original edge_index.
        """
        adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device), \
                    binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]
        # adj_mats[0]: identitiy, adj_mats[1]: adj_matrix
        
        for i in range(2, order+1):
            adj_mats.append(binarize(adj_mats[i-1] @ adj_mats[1]))
        # adj_mats[2] = adj_mats[1] @ adj_mats[1]: 2번 건너 도착하는 adj_mat
        # adj_mats[3] = adj_mats[2] @ adj_mats[1]: 3번 건너 도착하는 adj_mat

        order_mat = torch.zeros_like(adj)
        
        for i in range(1, order+1):
            order_mat += (adj_mats[i] - adj_mats[i-1]) * i
        
        return order_mat    # edge를 몇 번 타야되는지에 대한 정보가 담긴 adjacency matrix
    
    num_types = len(BOND_TYPES)
    
    N = num_nodes
    adj = to_dense_adj(edge_index).squeeze(0)   # (N, N)
    adj_order = get_higher_order_adj_matrix(adj, order) # (N, N)
    
    type_mat = to_dense_adj(edge_index, edge_attr=edge_type).squeeze(0) # (N, N)
    type_highorder = torch.where(adj_order > 1, num_types + adj_order - 1, torch.zeros_like(adj_order))
    # 22까지는 bond_type, 23 이상은 edge를 몇 번 타고 가냐에 대한 정보
    assert (type_mat * type_highorder == 0).all()
    type_new = type_mat + type_highorder
    
    new_edge_index, new_edge_type = dense_to_sparse(type_new)
    _, edge_order = dense_to_sparse(adj_order)
    
    # 중복 index 제거 및 cost addiction / N, N은 input sparse mat.의 shape
    new_edge_index, new_edge_type = coalesce(new_edge_index, new_edge_type.long(), N, N)
    
    return new_edge_index, new_edge_type

def _extend_to_radius_order(
    pos,
    edge_index,
    edge_type,
    cutoff,
    batch,
    unspecified_type_number=0,
    is_sidechain=None,
):
    # cutoff 내의 atom 간 연결(edge_index) 생성, distatnce는 계산 x
    assert edge_type.dim() == 1
    N = pos.size(0) # atom 개수
    
    bgraph_adj = torch.sparse.LongTensor(
        edge_index,
        edge_type,
        torch.Size([N, N])
    )

#    if is_sidechain is None:
    rgraph_edge_index = radius_graph(pos, r=cutoff, batch=batch)    # (2, E_r)
    
    rgraph_adj = torch.sparse.LongTensor(
        rgraph_edge_index,
        torch.ones(rgraph_edge_index.size(1)).long().to(pos.device) * unspecified_type_number,
        torch.Size([N, N]),
    )
    
    composed_adj = (bgraph_adj + rgraph_adj).coalesce() # Sparse (N, N)
    # edge_index = composed_adj.indices()
    # dist = (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)])
    
    new_edge_index = composed_adj.indices()
    new_edge_type = composed_adj.values().long()
    
    return new_edge_index, new_edge_type

def extend_graph_order_radius(
    num_nodes,
    pos,
    edge_index,
    edge_type,
    batch,
    order=3,
    cutoff=10.0,
    extend_order=True,
    extend_radius=True,
    is_sidechain=None,
):
    
    if extend_order:
        edge_index, edge_type = _extend_graph_order(
            num_nodes=num_nodes,
            edge_index=edge_index,
            edge_type=edge_type,
            order=order,
        )
    
    if extend_radius:
        edge_index, edge_type = _extend_to_radius_order(
            pos=pos,
            edge_index=edge_index,
            edge_type=edge_type,
            cutoff=cutoff,
            batch=batch,
            is_sidechain=is_sidechain,
        )
    
    return edge_index, edge_type