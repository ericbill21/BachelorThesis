import argparse

import numpy as np
import torch
import torch_geometric
import torch_geometric.transforms as T
from models import generic_gnn, generic_wlnn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from test_theoretical_acc import calculate_max_accuracy
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
from utils import Constant_Long, NormalizedDegree, Wrapper_WL_TUDataset, seed_everything

import wandb

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch GNN')
parser.add_argument('--model_path', type=str, default="")
parser.add_argument('--k_fold', type=int, default=10, help='Number of folds for k-fold cross validation.')
args = parser.parse_args()

# Load model
model = torch.load(args.model_path)
config = model.config
model.eval()

# Load dataset
seed_everything(config['seed'])
dataset_original = TUDataset(root="Code/datasets/", name=config['dataset'], use_node_attr=False).shuffle()

# Modify dataset
if isinstance(model, generic_wlnn):
    if dataset_original._data.x is None:
        print('No node features found. Using constant function for 1WL+NN.')
        dataset_original.transform = Constant_Long(0)

    dataset = Wrapper_WL_TUDataset(dataset_original, k_wl=config['k_wl'], wl_convergence=config['wl_convergence'])
    name = f"Aggregate: 1-WL+NN: {config['dataset']}"

elif isinstance(model, generic_gnn):
    if dataset_original._data.x is None:
        print('No node features found. Using one-hot degree for GNNs.')
        max_degree = 0
        degs = []
        for data in dataset_original:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset_original.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset_original.transform = NormalizedDegree(mean, std)

    dataset = dataset_original
    name = f"Aggregate: GNN: {config['dataset']}"

else:
    exit()

print(dataset.data.x)

wandb.init(
    project="BachelorThesisExperiments",
    name=name,
    config=config,
    tags=["aggregate_analysis"]
)

with torch.no_grad():
    # Collect all graph aggregates
    data_aggregate, data_y = [], []
    for data in dataset:

        if isinstance(model, generic_wlnn):
            x = model.embedding(data.x).squeeze()
            x = model.pool(x, data.batch).squeeze()

        if isinstance(model, generic_gnn):
            x = model.gnn(data.x, data.edge_index).squeeze()
            x = model.pool(x, data.batch).squeeze()

        data_aggregate.append(x)
        data_y.append(data.y)

    # Stack tensor to one big tensor
    data_aggregate = torch.stack(data_aggregate, dim=0)
    data_y = torch.stack(data_y, dim=0).squeeze()

    # Calculating max accuracy on aggregated data
    max_accuracy = calculate_max_accuracy(data_aggregate, data_y)
    print(f"Max accuracy: {max_accuracy}")
    wandb.log({"max_accuracy": max_accuracy})

    # Reduce Data to 2D
    # PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data_aggregate)
    reduced_data = torch.nn.functional.normalize(torch.tensor(reduced_data), p=2, dim=0)
    
    table = wandb.Table(data=[[reduced_data[i][0], reduced_data[i][1], data_y[i]] for i in range(len(reduced_data))],
                        columns = ["x", "y", "class"])
    wandb.log({"pca": table})
    
    # TSNE
    tsne = TSNE(n_components=2)
    reduced_data = tsne.fit_transform(data_aggregate)
    reduced_data = torch.nn.functional.normalize(torch.tensor(reduced_data), p=2, dim=0)

    table = wandb.Table(data=[[reduced_data[i][0], reduced_data[i][1], data_y[i]] for i in range(len(reduced_data))],
                        columns = ["x", "y", "class"])
    wandb.log({"tsne": table})

    # Testing different clustering algorithms
    wandb.define_metric("k")
    wandb.define_metric("mean_knn_acc", summary="max", step_metric="k")
    
    # Cross validation
    skf = KFold(n_splits=args.k_fold, shuffle=True)
    splits = list(skf.split(data_aggregate))

    num_test = 200

    # Clustering alogrithm: KNN
    knn_acc = np.zeros(num_test)
    for k in range(num_test):
    
        for fold, (train_index, test_index) in enumerate(splits):

            clustering_algorithm = KNeighborsClassifier(n_neighbors=k+1)
            clustering_algorithm.fit(data_aggregate[train_index], data_y[train_index])

            score = clustering_algorithm.score(data_aggregate[test_index], data_y[test_index])
            knn_acc[k] += score
        
        knn_acc[k] /= args.k_fold
        print(f"KNN accuracy for k={k+1}: {knn_acc[k]}")

    knn_table = wandb.Table(data=[[knn_acc[i], i+1] for i in range(len(knn_acc))], columns=["accuracy", "k"])
    wandb.log({"knn": knn_table})

    wandb.finish()