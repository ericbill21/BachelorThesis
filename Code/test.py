from torch_geometric.datasets import TUDataset

dataset = TUDataset(root=f'Code/datasets', name=f'PROTEINS', use_node_attr=False)
print(dataset.x.shape)
print(dataset.x)