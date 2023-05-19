import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import Constant, Compose
from utils import WL_Transformer
from utils import Wrapper_TUDataset

wl = WL_Transformer(use_node_attr=True, max_iterations=3, check_convergence=False, device=torch.device('cpu'))

dataset = Wrapper_TUDataset(k_wl=4, wl_convergence=False,
                            root=f'Code/datasets', name=f'PROTEINS', use_node_attr=True,
                            pre_transform=Compose([wl]), pre_shuffle=True)
print(dataset[0].x)

