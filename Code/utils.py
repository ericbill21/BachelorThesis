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

import os, shutil

from torch_geometric.datasets import TUDataset
from torch_geometric.io import read_tu_data
from typing import Callable, List, Optional

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
        max_iterations: int = -1,
        check_convergence: bool = False,
    ):
        self.wl_conv = wl_conv
        self.use_node_attr = use_node_attr
        self.max_iterations = max_iterations
        self.check_convergence = check_convergence

    def __call__(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
            
        # If there are no node features, we create a constant feature vector
        if data.x is None or not self.use_node_attr:
            data.x = torch.zeros((data.num_nodes, 1), dtype=torch.long)

        elif data.x.dim() > 1:
            data.x = data.x[:, 1:] #Remove first column #TODO: Check if this is correct

            assert (data.x.sum(dim=-1) == 1).sum() == data.x.size(0), 'Check if it is one-hot encoded'
            data.x = data.x.argmax(dim=-1)  # one-hot -> integer.:
        
        # Replace the graph features directly with the WL coloring
        if self.max_iterations == -1:
            self.max_iterations = data.num_nodes

        old_coloring = data.x.squeeze()
        new_coloring = self.wl_conv.forward(old_coloring, data.edge_index)

        iteration = 0     
        while ((not self.check_convergence) or (not check_wl_convergence(old_coloring, new_coloring))) and iteration < self.max_iterations:
            # Calculate the new coloring
            old_coloring = new_coloring
            new_coloring = self.wl_conv.forward(old_coloring, data.edge_index)

            iteration += 1

        data.x = old_coloring.unsqueeze(-1)

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'

def check_wl_convergence(old_coloring, new_coloring):
    mapping = {}

    for c in zip(old_coloring, new_coloring):
        if not c[0].item() in mapping:
            mapping[c[0].item()] = c[1].item()
        elif mapping[c[0].item()] != c[1].item():
            return False
        
    return True

class Wrapper_TUDataset(TUDataset):
    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 use_node_attr: bool = False, use_edge_attr: bool = False,
                 cleaned: bool = False,
                 pre_shuffle: bool = False):
        
        # Remove the processed folder if it exists
        if os.path.isdir(root + '/' + name + '/processed'):
            shutil.rmtree(root + '/' + name + '/processed')
        
        self.pre_shuffle = pre_shuffle
        super().__init__(root, name, transform, pre_transform,
                                        pre_filter, use_node_attr,
                                        use_edge_attr, cleaned)
    
    def process(self):
        self.data, self.slices, sizes = read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None or self.pre_transform is not None or self.pre_shuffle is not None:
            # Apply permutation to all attributes
            if self.pre_shuffle:
                data_list = [self.get(idx) for idx in torch.randperm(len(self))]
                
            # Create a list of Data objects
            else:
                data_list = [self.get(idx) for idx in range(len(self))]
            
            # Apply pre_filter
            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            # Apply pre_transform
            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            # Collate the list of Data objects into a single Data object
            self.data, self.slices = self.collate(data_list)
        
        sizes['num_node_labels'] = 1 # Hardcoded for now, TODO: make it more dynamic

        torch.save((self._data, self.slices, sizes), self.processed_paths[0])
