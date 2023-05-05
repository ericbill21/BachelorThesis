from typing import Callable, Optional
import torch
from torch import nn
from torch_geometric.nn.conv import wl_conv
from torch_geometric.datasets import TUDataset

class WL_TUDataset(TUDataset):

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 use_node_attr: bool = False, use_edge_attr: bool = False,
                 cleaned: bool = False, wl_transformer = None,
                 shuffle: bool = False) -> None:
        super().__init__(root, name, transform, pre_transform, pre_filter, use_node_attr, use_edge_attr, cleaned)


class WLNN(nn.Module):

    def __init__(self, f_enc = None, mlp = None) -> None:
        super().__init__()
        self.encoding = f_enc
        self.mlp = mlp

    def forward(self, data):
        data = self.encoding.forward(data)
        data = self.mlp(data)
        return data


    # def wl_transformation(self, dataset):
    #     self.wl.reset_parameters()

    #     new_dataset = torch.zeros((len(dataset), self.encoding.out_dim), dtype=torch.long)
    #     new_targets = torch.zeros((len(dataset)), dtype=torch.long)

    #     count = 0
    #     for graph in dataset:

    #         coloring = wl_algorithm(self.wl, graph)

    #         # We encode the node features
    #         new_dataset[count] = self.encoding.call(coloring)
    #         new_targets[count] = graph.y
    #         count += 1

    #     return torch.utils.data.TensorDataset(new_dataset, new_targets)
    
def create_transformer(wl):
    def constant_and_id_transformer(data):
        # If there are no node features, we create a constant feature vector
        if data.x is None:
            data.x = torch.zeros((data.edge_index.shape[1], 1), dtype=torch.long)
        
        # Replace the graph features directly with the WL coloring
        data.x = wl_algorithm(wl, data).unsqueeze(-1)
        
        return data
    
    return constant_and_id_transformer

def wl_algorithm(wl, graph, total_iterations = -1):
        if total_iterations == -1:
            total_iterations = graph.num_nodes

        old_coloring = graph.x.squeeze()
        new_coloring = wl.forward(old_coloring, graph.edge_index)

        iteration = 0
        is_converged = (torch.sort(wl.histogram(old_coloring))[0] == torch.sort(wl.histogram(new_coloring))[0]).all()
        while not is_converged and iteration < total_iterations:
            # Calculate the new coloring
            old_coloring = new_coloring
            new_coloring = wl.forward(old_coloring, graph.edge_index)

            # Check if the coloring has converged
            iteration += 1
            is_converged = (torch.sort(wl.histogram(old_coloring))[0] == torch.sort(wl.histogram(new_coloring))[0]).all()

        return old_coloring


    
        