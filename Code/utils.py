import os
import random
import shutil
import warnings
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data, HeteroData, dataset
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
    torch.mps.manual_seed(seed)

# Simple training loop
def train(model, loader, optimizer, loss_func, DEVICE):
    # Set model to training mode
    model.train()

    loss_all = 0
    correct = 0
    for data in loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()

        # Count the number of correct predictions and accumulate the loss
        pred = model(data.x, data.edge_index, data.batch)
        correct += (pred.max(1)[1] == data.y).sum().item()
        loss = loss_func(pred, data.y)

        # Update the weights
        loss.backward()
        optimizer.step()

        loss_all += data.num_graphs * loss.item()

    avg_loss = loss_all / len(loader.dataset)
    avg_acc = (correct / len(loader.dataset)) * 100

    return avg_loss, avg_acc
    

# Simple validation loop
def val(model, loader, loss_func, DEVICE, metric_func = {}):
    # Set model to evaluation mode
    model.eval()

    # Variable to store all predictions and all truth labels
    y_pred = torch.empty(0, device=DEVICE)
    y_true = torch.empty(0, device=DEVICE)

    # Variable to store the accumulated loss and the number of correct predictions
    loss_all = 0
    correct = 0
    for data in loader:
        data = data.to(DEVICE)

        # Compute the predictions and accumulate the loss
        pred_batch = model(data.x, data.edge_index, data.batch)
        loss_all += loss_func(pred_batch, data.y).item()

        # Convert the predictions from log-probabilities to class labels
        y_pred_batch = pred_batch.max(1)[1]
        correct += (y_pred_batch == data.y).sum().item()

        # Store the predictions and the truth labels
        y_pred = torch.cat([y_pred, y_pred_batch], dim=0)
        y_true = torch.cat([y_true, data.y], dim=0)

    # Compute the metrics
    metric_results = {}
    for metric_name, metric in metric_func.items():
        metric_results[metric_name] = metric(y_pred, y_true).item()

    avg_loss = loss_all / len(loader.dataset)
    avg_acc = (correct / len(loader.dataset)) * 100

    return avg_loss, avg_acc, metric_results

# Simple test loop
def test(model, loader, DEVICE):
    # Set model to evaluation mode
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(DEVICE)
        pred = model(data.x, data.edge_index, data.batch).max(1)[1]
        correct += (pred == data.y).sum().item()
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

class Wrapper_TUDataset(TUDataset):
    '''Wrapper Function for the TUDataset class from PyG.
    This wrapper allows to pre_shuffle the dataset before applying the pre_transform, and 
    it allows to automatically re-process the dataset if the pre_transform changes by deleting 
    the old processed data.
    '''
    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[List] = None,
                 pre_filter: Optional[Callable] = None,
                 use_node_attr: bool = False, 
                 use_edge_attr: bool = False,
                 cleaned: bool = False,
                 pre_shuffle: bool = False,
                 reprocess: bool = False):
        
        # Set the use_node_attr attribute globally
        self.use_node_attr = use_node_attr
        
        # Update root path if WL transformer is used
        if pre_transform is not None and isinstance(pre_transform[-1], WL_Transformer):
            self.k_wl = pre_transform[-1].max_iterations
            self.wl_convergence =  pre_transform[-1].check_convergence

            root = f'{root}/{"wl_convergence_true" if self.wl_convergence else "wl_convergence_false"}_{self.k_wl}'

            if pre_shuffle is False:
                warnings.warn('WARNING: The WL transformer is used but pre_shuffle is set to False. This will to unbalanced color histograms across all samples.')
        
        # Transform the pre_transform list into a Compose object
        if pre_transform is not None:
            pre_transform = Compose(pre_transform)
    
        # Check if the processed data that is available used the same pre_transform
        if os.path.isdir(root + '/' + name + '/processed'):

            # Otherwise we have to re-process the data and remove the old one
            f = os.path.join(root + '/' + name + '/processed', 'pre_transform.pt')
            if os.path.exists(f):
                if torch.load(f) != dataset._repr(pre_transform) or reprocess:
                    print('Re-processing dataset. To disable this behavior, remove the previous pre-processed dataset folder.')
                    shutil.rmtree(root + '/' + name + '/processed')
        
        # Setting new attributes and initializing the super class
        self.pre_shuffle = pre_shuffle
        super().__init__(root=root,
                            name=name,
                            transform=transform,
                            pre_transform=pre_transform,
                            pre_filter=pre_filter,
                            use_node_attr=True, # We remove extra node attributes in the process function
                            use_edge_attr=use_edge_attr,
                            cleaned=cleaned)

        if hasattr(self, 'num_node_features') and self.num_node_features > 0:
            self.max_node_feature = torch.max(self._data.x).item() 
    

    def process(self):
        self.data, self.slices, sizes = read_tu_data(self.raw_dir, self.name)

        # Remove additional continuous node attributes before applying the pre_transform
        if self._data.x is not None and not self.use_node_attr:
            num_node_attributes = sizes['num_node_attributes']
            self._data.x = self._data.x[:, num_node_attributes:]
    
        if self.pre_filter is not None or self.pre_transform is not None or self.pre_shuffle is not None:
            # Apply permutation to all attributes
            if self.pre_shuffle:
                data_list = [self.get(idx) for idx in torch.randperm(len(self))]
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
        
        sizes['num_node_labels'] = 1 
        torch.save((self._data, self.slices, sizes), self.processed_paths[0])