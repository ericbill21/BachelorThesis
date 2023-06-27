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
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import OneHotDegree, ToDevice
from torch_geometric.utils import degree
from utils import Constant_Long, NormalizedDegree, WL_Transformer, Wrapper_WL_TUDataset

import wandb

# GLOBAL VARIABLES
LOG_INTERVAL = 50
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
SAVE_MODEL = True
K_MAX = 150

# Parse arguments
parser = argparse.ArgumentParser(description='BachelorThesisExperiments')
parser.add_argument('--dataset', type=str, help='Dataset name.')
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
        "dataset": args.dataset,
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

# Load dataset from https://chrsmrrs.github.io/datasets/docs/datasets/.
dataset_original = TUDataset(root=f"Code/datasets", name=args.dataset, use_node_attr=False, pre_transform=ToDevice(DEVICE)).shuffle()

# In the case where no node features are available, we use one-hot degree for GNNs and the constant function for 1WL+NN.
# The following if clause is inspired from  https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/datasets.py.
if dataset_original._data.x is None:
    if args.model.startswith("1WL+NN"):
        print('No node features found. Using constant function for 1WL+NN.')
        dataset_original.transform = Constant_Long(0)

    else:
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

# Lists for logging
test_accuracies = []
train_accuracies = []
val_accuracies = []
num_epochs = []

# Saving model
best_model_global = 0.0
best_models = []

# Testing Aggregation
if SAVE_MODEL:
    knn_accuracies = [[] for i in range(K_MAX)]
    svm_lin_accuracies = []
    svm_rbf_accuracies = []

# Number of repitions
for i in range(args.num_repition):
    print(f"Repition {i+1}/{args.num_repition}:")

    # Initialize k-fold cross validation
    kf = KFold(n_splits=args.k_fold, shuffle=True)
    dataset_original.shuffle()

    # Precalculate the Weisfeiler-Lehman coloring for the dataset.
    if args.model.startswith("1WL+NN"):
        dataset_current = Wrapper_WL_TUDataset(dataset_original, args.k_wl, args.wl_convergence, DEVICE)
        args.encoding_kwargs["max_node_feature"] = dataset_current.max_node_feature + 1

    else:
        dataset_current = dataset_original

    print(dataset_current[0].x)

    # Initialize lists for logging
    test_accuracies += [[]]
    train_accuracies += [[]]
    val_accuracies += [[]]

    splits = kf.split(list(range(len(dataset_current))))
    for fold, (train_index, test_index) in enumerate(splits):
        print(f"\tCross-Validation Split {fold+1}/{args.k_fold}")

        # Sample 10% split from training split for validation.
        train_index, val_index = train_test_split(train_index, test_size=0.1)

        # Variables for detemining the best model of the current fold.
        best_val_acc = 0.0

        # Split data.
        train_dataset = dataset_current[train_index.tolist()]
        val_dataset = dataset_current[val_index.tolist()]
        test_dataset = dataset_current[test_index.tolist()]
        
        # Prepare batching.
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

        # Setup model.
        model = create_model(model_name=args.model,
                                input_dim=dataset_current.num_node_features,
                                output_dim=dataset_current.num_classes,
                                mlp_kwargs=args.mlp_kwargs,
                                gnn_kwargs=args.gnn_kwargs,
                                encoding_kwargs=args.encoding_kwargs).to(DEVICE)
        model.reset_parameters()
        
        # Setup Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                factor=0.5, patience=5,
                                                                min_lr=0.0000001)
        for epoch in range(1, args.max_epochs + 1):
            if epoch % LOG_INTERVAL == 0: print(f"\t\tEpoch {epoch}/{args.max_epochs}")

            lr = scheduler.optimizer.param_groups[0]['lr']
            utils.train(train_loader, model, optimizer, DEVICE)
            val_acc = utils.test(val_loader, model, DEVICE) * 100.0
            scheduler.step(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = utils.test(test_loader, model, DEVICE) * 100.0
                best_train_acc = utils.test(train_loader, model, DEVICE) * 100.0

                # Save and override the current best model if the test accuracy is better than the current best.
                if SAVE_MODEL and best_test_acc > best_model_global:
                    data_aggregate = utils.get_agg_data(model, dataset_current)

                    best_model_global = best_test_acc
                    model.config = vars(args)
                    model.data_aggregate = data_aggregate
                    model.train_index = train_index
                    model.test_index = test_index
                    model.dataset = dataset_current
                    torch.save(model, f"Code/saved_models/Replicate_{run.name}.pt")

            # Break if learning rate is smaller 10**-6.
            if lr < 0.000001:
                print(f"\t\tEpoch {epoch}/{args.max_epochs}")
                break
        
        # AFTER TRAINING
        # Log results.
        test_accuracies[i].append(best_test_acc)
        train_accuracies[i].append(best_train_acc)
        val_accuracies[i].append(best_val_acc)
        num_epochs.append(epoch)

        # Test the aggregate of the best model.
        if SAVE_MODEL:

            for k in range(K_MAX):
                knn_acc = utils.test_knn(data_aggregate, train_index=train_index, test_index=test_index, k=k+1) * 100.0
                knn_accuracies[k].append(knn_acc)

            svm_acc_linear = utils.test_svm(data_aggregate, train_index=train_index, test_index=test_index, kernel='linear', max_iter=100000) * 100.0
            print(svm_acc_linear)
            svm_lin_accuracies.append(svm_acc_linear)

            svm_acc_rbf = utils.test_svm(data_aggregate, train_index=train_index, test_index=test_index, kernel='rbf', C=1.0, gamma='scale') * 100.0
            svm_rbf_accuracies.append(svm_acc_rbf)
        
# Transform to torch.Tensor
test_accuracies = torch.tensor(test_accuracies)
train_accuracies = torch.tensor(train_accuracies)
val_accuracies = torch.tensor(val_accuracies)
num_epochs = torch.tensor(num_epochs, dtype=torch.float32)

if SAVE_MODEL:
    wandb.define_metric('k')
    wandb.define_metric('knn_accuracies', step_metric='k')
    wandb.define_metric('knn_accuracies_std', step_metric='k')
    knn_accuracies = torch.tensor(knn_accuracies)

    for k in range(K_MAX):
        wandb.log({'k': k+1, 'knn_accuracies': knn_accuracies[k].mean(), 'knn_accuracies_std': knn_accuracies[k].std()})

    svm_lin_accuracies = torch.tensor(svm_lin_accuracies)
    svm_rbf_accuracies = torch.tensor(svm_rbf_accuracies)

    wandb.log({'svm_lin_accuracies': svm_lin_accuracies.mean(), 'svm_lin_accuracies_std': svm_lin_accuracies.std(),
                'svm_rbf_accuracies': svm_rbf_accuracies.mean(), 'svm_rbf_accuracies_std': svm_rbf_accuracies.std()})

print(f"Test Accuracies: {test_accuracies.mean()} with {test_accuracies.std()} std")
print(f"Train Accuracies: {train_accuracies.mean()} with {train_accuracies.std()} std")
print(f"Val Accuracies: {val_accuracies.mean()} with {val_accuracies.std()} std")
print(f"Number of epochs on average: {num_epochs.mean()} with {num_epochs.std()} std")

wandb.log({'test_accuracy': test_accuracies.mean(), 'test_accuracy_std': test_accuracies.std(),
           'train_accuracy': train_accuracies.mean(), 'train_accuracy_std': train_accuracies.std(),
           'val_accuracy' : val_accuracies.mean(), 'val_accuracy_std': val_accuracies.std(),
           'num_epochs': num_epochs.mean(), 'num_epochs_std': num_epochs.std()})
wandb.finish()