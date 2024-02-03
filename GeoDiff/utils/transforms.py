import torch

class CountNodesPerGraph(object):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data):
        data.num_nodes_per_graph = torch.LongTensor([data.num_nodes])
        return data
