from typing import Callable, Optional
import torch
from torch import nn
from torch_geometric.nn.conv import wl_conv
from torch_geometric.datasets import TUDataset

from typing import List, Optional, Union

import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('constant_long')
class Constant_Long(BaseTransform):
    r"""Appends a constant value to each node feature :obj:`x`
    (functional name: :obj:`constant`).

    Args:
        value (float, optional): The value to add. (default: :obj:`1.0`)
        cat (bool, optional): If set to :obj:`False`, existing node features
            will be replaced. (default: :obj:`True`)
        node_types (str or List[str], optional): The specified node type(s) to
            append constant values for if used on heterogeneous graphs.
            If set to :obj:`None`, constants will be added to each node feature
            :obj:`x` for all existing node types. (default: :obj:`None`)
    """
    def __init__(
        self,
        value: float = 1.0,
        cat: bool = True,
        node_types: Optional[Union[str, List[str]]] = None,
    ):
        if isinstance(node_types, str):
            node_types = [node_types]

        self.value = value
        self.cat = cat
        self.node_types = node_types

    def __call__(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:

        for store in data.node_stores:
            if self.node_types is None or store._key in self.node_types:
                num_nodes = store.num_nodes
                c = torch.full((num_nodes, 1), self.value, dtype=torch.long)

                if hasattr(store, 'x') and self.cat:
                    x = store.x.view(-1, 1) if store.x.dim() == 1 else store.x
                    store.x = torch.cat([x, c.to(x.device, x.dtype)], dim=-1)
                else:
                    store.x = c

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(value={self.value})'

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
    
def create_1wl_transformer(wl):

    def transformer(data):
        # If there are no node features, we create a constant feature vector
        if data.x is None:
            data.x = torch.zeros((data.num_nodes, 1), dtype=torch.long)
        
        # Replace the graph features directly with the WL coloring
        data.x = wl_algorithm(wl, data).unsqueeze(-1)
        
        return data
    
    return transformer

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


    
        