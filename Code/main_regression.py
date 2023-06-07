import argparse
import ast
import time

import numpy as np
import torch
import torch_geometric
import torch_geometric.transforms as T
import utils
from models import create_model
from sklearn.model_selection import KFold, train_test_split
from torch_geometric.data import Data, HeteroData, InMemoryDataset, dataset
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn.conv.wl_conv import WLConv
from torch_geometric.transforms import OneHotDegree, ToDevice
from torch_geometric.utils import degree
from utils import Constant_Long, NormalizedDegree, WL_Transformer, Wrapper_WL_TUDataset

import wandb

# GLOBAL VARIABLES
LOG_INTERVAL = 50
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Parse arguments
parser = argparse.ArgumentParser(description='BachelorThesisExperiments')
parser.add_argument('--max_epochs', type=int, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, help='Number of samples per batch.')
parser.add_argument('--lr', type=float, help='Initial learning rate.')
parser.add_argument('--k_fold', type=int, default=10, help='Number of folds for k-fold cross validation.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--k_wl', type=int, help='Number of Weisfeiler-Lehman iterations, or if -1 it runs until convergences.')
parser.add_argument('--model', type=str, default='1WL+NN:Embedding-Sum', help='Model to use. Options are "1WL+NN:Embedding-{SUM,MAX,MEAN}" or "GIN:{SUM,MAX,MEAN}".')
parser.add_argument('--wl_convergence', type=str, choices=['True','False'], help='Whether to use the convergence criterion for the Weisfeiler-Lehman algorithm.')
parser.add_argument('--tags', nargs='+', default=[], help='Tags for the run on wandb.')
parser.add_argument('--num_repition', type=int, default=1, help='Number of repitions.')
parser.add_argument('--transformer_kwargs', type=str, default='{}', help='Arguments for the transformer. For example, for the OneHotDegree transformer, the argument is the maximum degree.')
parser.add_argument('--encoding_kwargs', type=str, default='{}', help='Arguments for the encoding function. For example, for Embedding, the argument is the embedding dimension with the key "embedding_dim".')
parser.add_argument('--mlp_kwargs', type=str, default='{}', help='Arguments for the MLP. For example, for the MLP, the argument is the number of hidden layers with the key "num_layers".')
parser.add_argument('--gnn_kwargs', type=str, default='{}', help='Number of GNN layers.')
args = parser.parse_args()

# Convert arguments
args.wl_convergence = args.wl_convergence == "True"
args.encoding_kwargs = ast.literal_eval(args.encoding_kwargs)
args.gnn_kwargs = ast.literal_eval(args.gnn_kwargs)
args.mlp_kwargs = ast.literal_eval(args.mlp_kwargs)
args.transformer_kwargs = ast.literal_eval(args.transformer_kwargs)

# Set seed for reproducibility
utils.seed_everything(args.seed)

# Initialize wandb
run = wandb.init(
    project="BachelorThesisExperiments",
    name=f"{args.model}: {time.strftime('%d.%m.%Y %H:%M:%S')}",
    tags=args.tags,
    config={
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "device": DEVICE,
        "k_fold": args.k_fold,
        "dataset": "ZINC",
        "lr": args.lr,
        "seed": args.seed,
        "k_wl": args.k_wl,
        "model": args.model,
        "wl_convergence": args.wl_convergence} | args.encoding_kwargs | args.gnn_kwargs | args.mlp_kwargs | args.transformer_kwargs
)

# Define metrics
wandb.define_metric("test_accuracies")
wandb.define_metric("test_accuracies_std")
wandb.define_metric("train_accuracies")
wandb.define_metric("train_accuracies_std")
wandb.define_metric("val_accuracies")
wandb.define_metric("val_accuracies_std")
wandb.define_metric("num_epochs")
wandb.define_metric("num_epochs_std")

class Wrapper_Dataset(InMemoryDataset):
    def __init__(self, data_list: list):
        super().__init__(root=None, transform=None, pre_transform=None, pre_filter=None, log=None)
        self.data, self.slices = self.collate(data_list)
        self.max_node_feature = self._data.x.max().item()

# Load dataset from https://chrsmrrs.github.io/datasets/docs/datasets/.
zinc_test = TUDataset(root=f"Code/datasets", name="ZINC_test", use_node_attr=False, pre_transform=ToDevice(DEVICE)).shuffle()
zinc_train = TUDataset(root=f"Code/datasets", name="ZINC_train", use_node_attr=False, pre_transform=ToDevice(DEVICE)).shuffle()
zinc_val = TUDataset(root=f"Code/datasets", name="ZINC_val", use_node_attr=False, pre_transform=ToDevice(DEVICE)).shuffle()

# Lists for logging
test_losses = []
train_losses = []
val_losses = []
num_epochs = []

# Number of repitions
for i in range(args.num_repition):
    print(f"Repition {i+1}/{args.num_repition}:")

    zinc_test.shuffle()
    zinc_train.shuffle()
    zinc_val.shuffle()

    # Precalculate the Weisfeiler-Lehman coloring for the dataset.
    if args.model.startswith("1WL+NN"):
        wl_conv = WLConv().to(DEVICE)

        train_list = []
        test_list = []
        val_list = []

        # We iterate over each graph randomly in the concatenated dataset of train, val and test.
        for idx in torch.randperm(zinc_train.len() + zinc_val.len() + zinc_test.len()):
            idx = idx.item()

            # We need to check if the index is in the train, val or test set.
            if idx < zinc_train.len():
                train_list.append(zinc_train[idx])
                for _ in range(args.k_wl):
                    train_list[-1].x = wl_conv(train_list[-1].x.squeeze(), train_list[-1].edge_index)
                
            elif idx >= zinc_train.len() and idx < zinc_train.len() + zinc_val.len():
                val_list.append(zinc_val[idx - zinc_train.len()])
                for _ in range(args.k_wl):
                    val_list[-1].x = wl_conv(val_list[-1].x.squeeze(), val_list[-1].edge_index)

            elif idx >= zinc_train.len() + zinc_val.len() and idx < zinc_train.len() + zinc_val.len() + zinc_test.len():
                test_list.append(zinc_test[idx - zinc_train.len() - zinc_val.len()])
                for _ in range(args.k_wl):
                    test_list[-1].x = wl_conv(test_list[-1].x.squeeze(), test_list[-1].edge_index)
            
            else:
                raise ValueError("Index out of range.")
        
        train_dataset = Wrapper_Dataset(train_list)
        val_dataset = Wrapper_Dataset(val_list)
        test_dataset = Wrapper_Dataset(test_list)

        args.encoding_kwargs["max_node_feature"] = max(train_dataset.max_node_feature, val_dataset.max_node_feature, test_dataset.max_node_feature) + 1

    else:
        train_dataset = zinc_train
        val_dataset = zinc_val
        test_dataset = zinc_test

    print(train_dataset[0].x)

    # Initialize constant
    best_val_loss = float("inf")

    # Initialize lists for logging
    test_losses += [[]]
    train_losses += [[]]
    val_losses += [[]]
    
    # Prepare batching.
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # Setup model.
    model = create_model(model_name=args.model,
                            input_dim=train_dataset.num_node_features,
                            output_dim=1,
                            mlp_kwargs=args.mlp_kwargs,
                            gnn_kwargs=args.gnn_kwargs,
                            encoding_kwargs=args.encoding_kwargs,
                            is_classification=False).to(DEVICE)
    model.reset_parameters()
    
    # Setup Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                            factor=0.5, patience=5,
                                                            min_lr=0.0000001)
    for epoch in range(1, args.max_epochs + 1):
        if epoch % LOG_INTERVAL == 0: print(f"\t\tEpoch {epoch}/{args.max_epochs}")

        lr = scheduler.optimizer.param_groups[0]['lr']
        utils.train_regression(train_loader, model, optimizer, DEVICE)
        val_loss = utils.test_regression(val_loader, model, DEVICE)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_test_loss = utils.test_regression(test_loader, model, DEVICE)
            best_train_loss = utils.test_regression(train_loader, model, DEVICE)

        # Break if learning rate is smaller 10**-6.
        if lr < 0.000001:
            print(f"\t\tEpoch {epoch}/{args.max_epochs}")
            break

    test_losses[i].append(best_test_loss)
    train_losses[i].append(best_train_loss)
    val_losses[i].append(best_val_loss)
    num_epochs.append(epoch)
    
# Transform to torch.Tensor
test_losses = torch.tensor(test_losses)
train_losses = torch.tensor(train_losses)
val_losses = torch.tensor(val_losses)
num_epochs = torch.tensor(num_epochs, dtype=torch.float32)

print(f"Test Accuracies: {test_losses.mean()} with {test_losses.std()} std")
print(f"Train Accuracies: {train_losses.mean()} with {train_losses.std()} std")
print(f"Val Accuracies: {val_losses.mean()} with {val_losses.std()} std")
print(f"Number of epochs on average: {num_epochs.mean()} with {num_epochs.std()} std")

wandb.log({'test_accuracy': test_losses.mean(), 'test_accuracy_std': test_losses.std(),
           'train_accuracy': train_losses.mean(), 'train_accuracy_std': train_losses.std(),
           'val_accuracy' : val_losses.mean(), 'val_accuracy_std': val_losses.std(),
           'num_epochs': num_epochs.mean(), 'num_epochs_std': num_epochs.std()})
wandb.finish()