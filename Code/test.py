import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import ToDevice
from torch_geometric.utils import degree
from utils import Constant_Long, WL_Transformer, Wrapper_WL_TUDataset, seed_everything


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


# ENZYMES, NCI1  work flawlessly


# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# dataset = TUDataset(root=f"Code/test/datasets", name="IMDB-Multi", use_node_attr=False, pre_transform=ToDevice(DEVICE)).shuffle()

# if dataset.data.x is None:
#     max_degree = 0
#     degs = []
#     for data in dataset:
#         degs += [degree(data.edge_index[0], dtype=torch.long)]
#         max_degree = max(max_degree, degs[-1].max().item())

#     if max_degree < 1000:
#         #dataset.transform = T.OneHotDegree(max_degree)
#         #dataset.transform = T.Compose([Constant_Long(1), Constant_Long(0)])
#         dataset.transform = Constant_Long(0)

#     else:
#         deg = torch.cat(degs, dim=0).to(torch.float)
#         mean, std = deg.mean().item(), deg.std().item()
#         dataset.transform = NormalizedDegree(mean, std)


# dataset_rep = Wrapper_WL_TUDataset(dataset, 1, False)
# print('Hey')

x = torch.tensor([[1,2], [3,4], [1,2], [5,6]])
y = torch.tensor([0, 1, 1, 1])

import numpy as np
import torch


def calculate_max_accuracy(x, y):
    # Combine x and y into a single dataset
    dataset = torch.cat((x, y.unsqueeze(1)), dim=1)

    # Get unique samples in x
    unique_samples, unique_indices = torch.unique(x, dim=0, return_inverse=True)

    max_correct = 0
    total_samples = 0

    for i in range(unique_samples.shape[0]):
        # Find indices of matching samples in the dataset. 
        # Necessary to use flatten() to get a 1D tensor such that 'dataset[matching_indices]' returns a 2D tensor.
        matching_indices = torch.nonzero(unique_indices == i, as_tuple=False).flatten()

        # Get matching samples and labels
        matching_samples = dataset[matching_indices]
        matching_labels = matching_samples[:, -1]

        # Count the occurrences of each class label
        _, label_counts = torch.unique(matching_labels, return_counts=True)

        # Update the maximum correct count
        max_correct += torch.max(label_counts)

        # Update the total number of samples
        total_samples += matching_samples.shape[0]

    # Calculate the maximal accuracy
    max_accuracy = max_correct / total_samples

    return max_accuracy


print(calculate_max_accuracy(x,y))
print('hey')