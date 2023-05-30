import os
import random
import shutil
import warnings
from typing import Callable, List, Optional, Union

import numpy as np
import torch
import torch_geometric
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data, HeteroData, InMemoryDataset, dataset
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.datasets import TUDataset
from torch_geometric.io import read_tu_data
from torch_geometric.nn.conv import wl_conv
from torch_geometric.nn.conv.wl_conv import WLConv
from torch_geometric.transforms import BaseTransform, Compose


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# One training epoch for GNN model.
def train(train_loader, model, optimizer, device):
    model.train()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        optimizer.step()


# Get acc. of GNN model.
def test(loader, model, device):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


@functional_transform('constant_long')
class Constant_Long(BaseTransform):
    r"""Appends a constant value to each node feature :obj:`x`
    (functional name: :obj:`constant`).

    Args:
        value (int, optional): The value to add. (default: :obj:`1.0`)
        cat (bool, optional): If set to :obj:`False`, existing node features
            will be replaced. (default: :obj:`True`)
        node_types (str or List[str], optional): The specified node type(s) to
            append constant values for if used on heterogeneous graphs.
            If set to :obj:`None`, constants will be added to each node feature
            :obj:`x` for all existing node types. (default: :obj:`None`)
    """
    def __init__(
        self,
        value: int = 1.0,
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
        use_node_attr: bool = True,
        max_iterations: int = -1,
        check_convergence: bool = False,
        device: str = 'cpu',
    ):
        self.wl_conv = WLConv().to(device)
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
            assert (data.x.sum(dim=-1) == 1).sum() == data.x.size(0), 'Check if it is one-hot encoded'
            data.x = data.x.argmax(dim=-1)  # one-hot -> integer.:
        
        # If the max iterations is set to -1, we set it to the number of nodes
        if self.max_iterations == -1:
            self.max_iterations = data.num_nodes

        # 1-WL Algorithm
        old_coloring = data.x.squeeze()
        new_coloring = self.wl_conv.forward(old_coloring, data.edge_index)

        iteration = 0     
        while ((not self.check_convergence) or (not check_wl_convergence(old_coloring, new_coloring))) and iteration < self.max_iterations:
            old_coloring = new_coloring
            new_coloring = self.wl_conv.forward(old_coloring, data.edge_index)

            iteration += 1

        data.x = old_coloring.unsqueeze(-1)

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(use_node_attr={self.use_node_attr}, max_iterations={self.max_iterations}, check_convergence={self.check_convergence})'

    def get_largest_color(self):
        return len(self.wl_conv.hashmap)

def check_wl_convergence(old_coloring, new_coloring):
    mapping = {}

    for c in zip(old_coloring, new_coloring):
        if not c[0].item() in mapping:
            mapping[c[0].item()] = c[1].item()
        elif mapping[c[0].item()] != c[1].item():
            return False
        
    return True

class Wrapper_WL_TUDataset(InMemoryDataset):
    def __init__(self, dataset: torch_geometric.datasets, k_wl: int, wl_convergence: bool, DEVICE: str):
        super().__init__(root=None, transform=None, pre_transform=None, pre_filter=None, log=None)

        # First we copy the given Dataset with respect to the current ordering
        print("Copying dataset...")
        data_list = []
        for idx in range(dataset.len()):
            data_list.append(dataset[idx])
        
        print(f"Device: {data_list[0].x.device}")
        print("Applying WL Algorithm...")

        # Apply k_wl times the 1-WL convolution
        self.wl_conv = WLConv().to(DEVICE)
        for data in data_list:
            for _ in range(k_wl):
                data.x = self.wl_conv(data.x, data.edge_index)

        print("Applying WL Algorithm... Done")

        self.data, self.slices = self.collate(data_list)
        print("collate done")
        print(f"device {self._data.x.device}")


        # Make the array range small for smaller embedding later on
        self._data.x = self._data.x - self._data.x.min()

        # Save the maximum node feature as attribute
        self.max_node_feature = self._data.x.max().item()