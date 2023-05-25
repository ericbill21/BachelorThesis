import argparse

import numpy as np
import torch
import torch_geometric
from models import generic_gnn, generic_wlnn
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from torch_geometric.datasets import TUDataset
from utils import WL_Transformer, Wrapper_TUDataset, seed_everything

import wandb

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch GNN')
parser.add_argument('--model_path', type=str, default="")
args = parser.parse_args()

# Load model
model = torch.load(args.model_path)
model.eval()

DATASET_NAME = model.dataset_config["dataset_name"]
K_WL = model.dataset_config["k_wl"]
WL_CONVERGENCE = model.dataset_config["wl_convergence"]
TRANSFORMER = model.dataset_config["transformer"]
TRANSFORMER_ARGS = model.dataset_config["transformer_args"]
SEED = model.dataset_config["seed"]

# Initialize wandb
model_name = args.model_path.split("/")[-1]
wandb.init(
    project="BachelorThesis",
    name=f"Test Aggregates: {model_name}",
    tags=["Aggregate Test"],
    config={
        "dataset_name": DATASET_NAME,
        "seed": SEED,
        "k_wl": K_WL,
        "wl_convergence": WL_CONVERGENCE,
    }
)

# Set seed
seed_everything(SEED)

# Load dataset
if isinstance(model, generic_wlnn):
    transformer = [WL_Transformer(
            use_node_attr=True,
            max_iterations=K_WL,
            check_convergence=WL_CONVERGENCE,
        )]
else:
    transformer = None

dataset = Wrapper_TUDataset(root=f'Code/datasets', name=DATASET_NAME, use_node_attr=False, pre_shuffle=True, pre_transform=transformer, reprocess=True)

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

    # Normalize data
    data_aggregate = torch.nn.functional.normalize(data_aggregate, p=2, dim=0)

    # Get unique values for each class
    unique_values_per_class = {}
    for y in range(dataset.num_classes):
        aggregate_class = data_aggregate[data_y == y]
        unique_values = torch.unique(aggregate_class, dim=0, sorted=False).shape[0]
        
        unique_values_per_class[y] = unique_values
        wandb.summary[f'unique_aggregate_values: class {y}'] = unique_values

    # Reduce Data to 2D
    # PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data_aggregate)
    
    table = wandb.Table(data=[[reduced_data[i][0], reduced_data[i][1], data_y[i]] for i in range(len(reduced_data))],
                        columns = ["x", "y", "class"])
    wandb.log({"pca": table})
    
    # TSNE
    tsne = TSNE(n_components=2)
    reduced_data = tsne.fit_transform(data_aggregate)

    table = wandb.Table(data=[[reduced_data[i][0], reduced_data[i][1], data_y[i]] for i in range(len(reduced_data))],
                        columns = ["x", "y", "class"])
    wandb.log({"tsne": table})

    # Clustering alogrithm: KNN
    wandb.define_metric("k")
    wandb.define_metric("knn_acc", summary="max", step_metric="k")

    num_test = dataset.len() // 10
    knn_acc = np.zeros(num_test)
    for k in range(num_test):
        clustering_algorithm = KNeighborsClassifier(n_neighbors=k+1)
        clustering_algorithm.fit(data_aggregate, data_y)

        score = clustering_algorithm.score(data_aggregate, data_y)
        knn_acc[k] = score
        wandb.log({"knn_acc": score,
                   "k" : k+1})

    best_k = np.argmax(knn_acc)
    print(f"KNN: {k+1} -> {knn_acc[best_k]}")