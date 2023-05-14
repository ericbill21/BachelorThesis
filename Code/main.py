import time
import numpy as np
import pandas as pd
import utils
import argparse

import wandb

import torch
import torch_geometric

import torch.nn as TorchNN
from torch_geometric.datasets import TUDataset
from utils import Constant_Long, WL_Transformer
from torch_geometric.transforms import OneHotDegree, ToDevice
from torch_geometric.nn.conv.wl_conv import WLConv
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader as TorchDataLoader

from torch_geometric.nn.models import GIN, MLP
from torch_geometric.nn import pool as PyGPool
from torch_geometric.nn import aggr as PyGAggr

from sklearn.model_selection import KFold, StratifiedKFold

import visualization

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch GNN')
parser.add_argument('--dataset', type=str, default='PROTEINS', help='Dataset name')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=32, help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=0.02, help='Initial learning rate.')
parser.add_argument('--k_fold', type=int, default=10, help='Number of folds for k-fold cross validation.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--k_wl', type=int, default=1, help='Number of Weisfeiler-Lehman iterations, or if -1 it runs until convergences.')
parser.add_argument('--model', type=str, default='1WL+NN:Embedding-Sum', help='Model to use.')
args = parser.parse_args()


# GLOBAL PARAMETERS
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
K_FOLD = args.k_fold
LEARNING_RATE = args.lr
DATASET_NAME = args.dataset
SEED = args.seed
K_WL = args.k_wl
MODEL_NAME = args.model

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
LOG_INTERVAL = 5
PLOT_RESULTS = True
NUM_EPOCHS_TO_BE_PRINTED = 5

wandb.init(
    # set the wandb project where this run will be logged
    project="BachelorThesis",

    name=f"{MODEL_NAME}: {time.strftime('%d.%m.%Y %H:%M:%S')}",
    
    # track hyperparameters and run metadata
    config={
    "Epochs": EPOCHS,
    "Batch size": BATCH_SIZE,
    "Device": DEVICE,
    "k-fold": K_FOLD,
    "Dataset": DATASET_NAME,
    "learning_rate": LEARNING_RATE,
    "seed": SEED,
    "k_wl": K_WL,
    }
)

wandb.define_metric("epoch")
wandb.define_metric("train accuracy: fold*", summary="max", step_metric="epoch")
wandb.define_metric("val accuracy: fold*", summary="max", step_metric="epoch")
wandb.define_metric("train loss: fold*", summary="min", step_metric="epoch")
wandb.define_metric("val loss: fold*", summary="min", step_metric="epoch")

# Simple training loop
def train(model, loader, optimizer, loss_func):
    # Set model to training mode
    model.train()

    loss_all = 0
    correct = 0
    for data in loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()

        # Make prediction
        pred = model(data.x, data.edge_index, data.batch)

        # Count the number of correct predictions
        correct += (pred.max(1)[1] == data.y).sum().item()

        # Calculate the loss and backpropagate
        loss = loss_func(pred, data.y)
        loss.backward()

        # Update the weights
        optimizer.step()

        loss_all += data.num_graphs * loss.item()

    return loss_all / len(loader.dataset), (correct / len(loader.dataset))*100

# Simple validation loop
def val(model, loader, loss_func):
    # Set model to evaluation mode
    model.eval()

    loss_all = 0
    correct = 0
    for data in loader:
        data = data.to(DEVICE)

        # Make prediction
        pred = model(data.x, data.edge_index, data.batch)

        # Count the number of correct predictions
        correct += (pred.max(1)[1] == data.y).sum().item()

        # Calculate the loss
        loss_all += loss_func(pred, data.y).item()

    return loss_all / len(loader.dataset), (correct / len(loader.dataset))*100

# Simple test loop
def test(model, loader):
    # Set model to evaluation mode
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(DEVICE)
        pred = model(data.x, data.edge_index, data.batch).max(1)[1]
        correct += (pred == data.y).sum().item()
    return correct / len(loader.dataset)

# Set seed for reproducibility
utils.seed_everything(SEED)

# Global wl convolution
wl = WLConv()

# Dataset from https://chrsmrrs.github.io/datasets/docs/datasets/
dataset = TUDataset(root=f'Code/datasets/{DATASET_NAME}', name=f'{DATASET_NAME}', 
                    pre_transform=ToDevice(DEVICE))
dataset = dataset.shuffle()

# Initialize dataset transformer #TODO: check that
max_degree = max([data.num_nodes for data in dataset])
one_hot_degree_transformer = OneHotDegree(max_degree=max_degree)

# Split dataset into K_FOLD folds
cross_validation = StratifiedKFold(n_splits=K_FOLD, shuffle=True, random_state=SEED)

# Initialize the model
if MODEL_NAME == "1WL+NN:Embedding-Sum":
    # TODO: make it more elegant
    dataset.transform = WL_Transformer(wl, use_node_attr=True)
    for data in dataset:
        pass
    largest_color = len(wl.hashmap)

    # Check how many iterations the WL algorithm will run
    # If K_WL is -1, it will run until convergence, otherwise it will run K_WL iterations
    if K_WL == -1:
        dataset.transform = WL_Transformer(wl, use_node_attr=True)
        wl_conv_layers = []
    else:
        dataset.transform = Constant_Long(0) 
        wl_conv_layers = [(wl, 'x, edge_index -> x') for _ in range(K_WL)]
    
    # Initialize the model
    model = torch_geometric.nn.Sequential('x, edge_index, batch', wl_conv_layers + [
                    (TorchNN.Embedding(num_embeddings=largest_color, embedding_dim=10), 'x -> x'),
                    (torch.squeeze, 'x -> x'),
                    (PyGPool.global_add_pool, 'x, batch -> x'),
                    (MLP(channel_list=[10, 64, 32, 16, dataset.num_classes]), 'x -> x'),
                    (TorchNN.Softmax(dim=1), 'x -> x')
                ]).to(DEVICE)

else :
    raise ValueError("Invalid model name")

# Log the model to wandb
wandb.watch(model, log="all")

# TRAINING LOOP
# Initialize the optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
loss_func = lambda pred, true: TorchNN.CrossEntropyLoss()(pred, true).log()

# Training Loop
runtime = []

# Loop over the K_FOLD splits
for fold, (train_ids, test_ids) in enumerate(cross_validation.split(dataset, dataset.y)):
    print(f'Cross-Validation Split {fold+1}/{K_FOLD}:')

    # Reset the model parameters
    model.reset_parameters()

    # Initialize the data loaders
    train_loader = PyGDataLoader(dataset[train_ids], batch_size=BATCH_SIZE, shuffle=True)
    val_loader = PyGDataLoader(dataset[test_ids], batch_size=BATCH_SIZE, shuffle=False)

    # Train the model
    for epoch in range(EPOCHS):
        start = time.time()

        # Train, validate and test the model
        train_loss, train_acc = train(model, train_loader, optimizer, loss_func)
        val_loss, val_acc = val(model, val_loader, loss_func)

        # Log the results to wandb
        wandb.log({f"train accuracy: fold {fold+1}": train_acc, f"val accuracy: fold {fold+1}": val_acc,
                  f"train loss: fold {fold+1}": train_loss, f"val loss: fold {fold+1}": val_loss, "epoch": epoch+1})

        # Print current status
        if (epoch + 1) % LOG_INTERVAL == 0:
            print(f'\tEpoch: {epoch+1},\t Train Loss: {round(train_loss, 5)},' \
                    f'\t Train Acc: {round(train_acc, 1)}%,\t Val Loss: {round(val_loss, 5)},' \
                    f'\t Val Acc: {round(val_acc, 1)}%')
        
        runtime.append(time.time()-start)
    
print(f'Avg runtime per epoch: {np.mean(runtime)}')


wandb.finish()
