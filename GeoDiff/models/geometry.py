import torch
from torch_scatter import scatter_add

def get_distance(pos, edge_index):
    # tensor.norm(-1)로 텐서의 크기 계산이 가능
    return (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)

def eq_transform(score_d, pos, edge_index, edge_length):
    N = pos.size(0)
    dd_dr = (1. / edge_length) * (pos[edge_index[0]] - pos[edge_index[1]])  # (E, 3)
    # scatter_add: 같은 인덱스끼리는 더해줌
    score_pos = scatter_add(dd_dr * score_d, edge_index[0], dim=0, dim_size=N) \
        + scatter_add(-dd_dr * score_d, edge_index[1], dim=0, dim_size=N)
    return score_pos
