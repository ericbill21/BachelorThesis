import os
import random
import shutil
import warnings
from typing import Callable, List, Optional, Union

import models
import numpy as np
import torch
import torch_geometric
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data, HeteroData, InMemoryDataset, dataset
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.datasets import TUDataset
from torch_geometric.io import read_tu_data
from torch_geometric.nn.conv import wl_conv
from torch_geometric.nn.conv.wl_conv import WLConv
from torch_geometric.transforms import BaseTransform, Compose
from torch_geometric.utils import degree


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# One training epoch for GNN model for a classification task.
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

# One training epoch for GNN model for a regression task.
def train_regression(train_loader, model, optimizer, device):
    model.train()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output.squeeze(), data.y, reduction="mean")
        loss.backward()
        optimizer.step()

# Get loss of GNN model.
def test_regression(loader, model, device):
    model.eval()

    loss = 0.0
    for data in loader:
        data = data.to(device)
        output = model(data)
        loss += F.mse_loss(output.squeeze(), data.y, reduction="sum")

    return loss / len(loader.dataset)

def calc_shannon_diversity(dataset):
    # Shannon diversity index for assessing class imbalance of a dataset.
    n = dataset.len()
    k = dataset.num_classes

    n, k = torch.tensor(n), torch.tensor(k)
    H = torch.tensor(0.0)
    for i in range(k):
        c_i = torch.count_nonzero(dataset._data.y == i)
        H -= (c_i / n) * torch.log2(c_i / n)

    unbalance = H / torch.log2(k)
    return unbalance.round(decimals=3).item()


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


def weisfeiler_leman(data, wl_conv): 
    data.x = data.x.squeeze()

    if data.x.dim() > 1:
        assert (data.x.sum(dim=-1) == 1).sum() == data.x.size(0), 'Check if it is one-hot encoded'
        data.x = data.x.argmax(dim=-1)  # one-hot -> integer.

    # 1-WL Algorithm
    old_coloring = data.x.squeeze()
    new_coloring = wl_conv.forward(old_coloring, data.edge_index)

    iteration = 0
    max_iterations = data.num_nodes  
    while (not check_wl_convergence(old_coloring, new_coloring)) and iteration < max_iterations:
        old_coloring = new_coloring
        new_coloring = wl_conv.forward(old_coloring, data.edge_index)

        iteration += 1

    data.x = old_coloring.unsqueeze(-1)

    return data


def check_wl_convergence(old_coloring, new_coloring):
    mapping = {}

    for c in zip(old_coloring, new_coloring):
        if not c[0].item() in mapping:
            mapping[c[0].item()] = c[1].item()
        elif mapping[c[0].item()] != c[1].item():
            return False
        
    return True

class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data

class Wrapper_WL_TUDataset(InMemoryDataset):
    def __init__(self, dataset: torch_geometric.datasets, k_wl: int, wl_convergence: bool, DEVICE: torch.device = torch.device("cpu")):
        super().__init__(root=None, transform=None, pre_transform=None, pre_filter=None, log=None)

        # First we copy the given Dataset with respect to the current ordering
        data_list = []
        for idx in range(len(dataset)):
            data_list.append(dataset[idx])
        
        # Apply k_wl times the 1-WL convolution
        self.wl_conv = WLConv().to(DEVICE)

        # This is the case for standard 1-WL 
        if k_wl == -1 and wl_convergence:
            print("Applying 1-WL convergence")
            for data in data_list:
                data = weisfeiler_leman(data, self.wl_conv)

        # Otherwise we apply k_wl times the 1-WL convolution
        else:
            for data in data_list:
                for _ in range(k_wl):
                    data.x = self.wl_conv(data.x.squeeze(), data.edge_index)

        self.data, self.slices = self.collate(data_list)

        # Make the array range small for smaller embedding later on
        self._data.x = self._data.x - self._data.x.min()

        # Save the maximum node feature as attribute
        self.max_node_feature = self._data.x.max().item()

def get_agg_data(model, dataset): 
    data_aggregate, data_y = [], []

    with torch.no_grad():
        for data in dataset:

            if isinstance(model, models.generic_wlnn):
                x = model.embedding(data.x).squeeze()
                x = model.pool(x, data.batch).squeeze()

            if isinstance(model, models.generic_gnn):
                x = model.gnn(data.x, data.edge_index).squeeze()
                x = model.pool(x, data.batch).squeeze()

            data_aggregate.append(x)
            data_y.append(data.y)

    # Stack tensor to one big tensor
    data_aggregate = torch.stack(data_aggregate, dim=0)
    data_y = torch.stack(data_y, dim=0).squeeze()

    data_aggregate = torch.cat([data_aggregate, data_y.unsqueeze(-1)], dim=-1)

    return data_aggregate

def test_knn(data_aggregate, train_index, test_index, k):
    X = data_aggregate[:,:-1]
    Y = data_aggregate[:,-1]

    clustering_algorithm = KNeighborsClassifier(n_neighbors=k)
    clustering_algorithm.fit(X[train_index], Y[train_index])

    score = clustering_algorithm.score(X[test_index], Y[test_index])
    return score

def test_svm(data_aggregate, train_index, test_index, **kwargs):
    X = data_aggregate[:,:-1]
    Y = data_aggregate[:,-1]
    
    clustering_algorithm = SVC(**kwargs)
    clustering_algorithm.fit(X[train_index], Y[train_index])

    score = clustering_algorithm.score(X[test_index], Y[test_index])
    return score
        

    

