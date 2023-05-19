import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import Constant, Compose
from utils import WL_Transformer
from utils import Wrapper_TUDataset

dataset = Wrapper_TUDataset(root=f'Code/datasets', name=f'PROTEINS', use_node_attr=False)

print('hey')