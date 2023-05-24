from torch_geometric.datasets import TUDataset
from utils import WL_Transformer, Wrapper_TUDataset

transformer = WL_Transformer(
    use_node_attr=True,
    max_iterations=-1,
    check_convergence=True,
)

dataset = Wrapper_TUDataset(root=f'Code/test_datasets', name=f'PROTEINS', use_node_attr=False, pre_shuffle=True)
print(dataset.x[0: 5])