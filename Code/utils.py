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

import random, os
import numpy as np
import torch

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.mps.manual_seed(seed)


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

@functional_transform('wl_alogorithm')
class WL_Transformer(BaseTransform):

    def __init__(
        self,
        wl_conv: torch.nn.Module,
        use_node_attr: bool = True,
    ):
        self.wl_conv = wl_conv
        self.use_node_attr = use_node_attr

    def __call__(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        
        # Check if the node features are one-hot encoded
        if data.x.dim() > 1:
            assert (data.x.sum(dim=-1) == 1).sum() == data.x.size(0), 'Check if it is one-hot encoded'
            data.x = data.x.argmax(dim=-1)  # one-hot -> integer.
        assert data.x.dtype == torch.long

        # If there are no node features, we create a constant feature vector
        if data.x is None or not self.use_node_attr:
            data.x = torch.zeros((data.num_nodes, 1), dtype=torch.long)
        
        # Replace the graph features directly with the WL coloring
        data.x = wl_algorithm(self.wl_conv, data).unsqueeze(-1)
        
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'


def wl_algorithm(wl, graph, total_iterations = -1):
        if total_iterations == -1:
            total_iterations = graph.num_nodes

        old_coloring = graph.x.squeeze()
        new_coloring = wl.forward(old_coloring, graph.edge_index)

        iteration = 0        
        while not check_wl_convergence(old_coloring, new_coloring) and iteration < total_iterations:
            # Calculate the new coloring
            old_coloring = new_coloring
            new_coloring = wl.forward(old_coloring, graph.edge_index)

            iteration += 1

        return old_coloring

def check_wl_convergence(old_coloring, new_coloring):
    mapping = {}

    for c in zip(old_coloring, new_coloring):
        if not c[0].item() in mapping:
            mapping[c[0].item()] = c[1].item()
        elif mapping[c[0].item()] != c[1].item():
            return False
        
    return True


    
        