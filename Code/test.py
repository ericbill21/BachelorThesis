import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import ToDevice
from utils import Wrapper_WL_TUDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dataset = TUDataset(root=f"Code/datasets", name="PROTEINS", use_node_attr=False, pre_transform=ToDevice(DEVICE)).shuffle()

dataset_rep = Wrapper_WL_TUDataset(dataset, 3, False)
