import torch

def get_distance(pos, edge_index):
    # tensor.norm(-1)로 텐서의 크기 계산이 가능
    return (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)