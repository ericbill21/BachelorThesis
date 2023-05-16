import time
import numpy as np
import utils
import argparse

import wandb

import torch
import torch_geometric

import torch.nn
from torch_geometric.datasets import TUDataset
from utils import Constant_Long, WL_Transformer
from torch_geometric.transforms import OneHotDegree, ToDevice
from torch_geometric.nn.conv.wl_conv import WLConv
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.nn.models import GIN, MLP

from sklearn.model_selection import StratifiedKFold

from utils import Wrapper_TUDataset

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch GNN')
parser.add_argument('--dataset', type=str, default='PROTEINS', help='Dataset name')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=32, help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=0.02, help='Initial learning rate.')
parser.add_argument('--k_fold', type=int, default=10, help='Number of folds for k-fold cross validation.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--k_wl', type=int, default=3, help='Number of Weisfeiler-Lehman iterations, or if -1 it runs until convergences.')
parser.add_argument('--model', type=str, default='1WL+NN:Embedding-Sum', help='Model to use.')
parser.add_argument('--wl_convergence', type=bool, default=False, action=argparse.BooleanOptionalAction, help='Whether to use the convergence criterion for the Weisfeiler-Lehman algorithm.')
parser.add_argument('--tags', nargs='+', default=[])
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
WL_CONVERGENCE = args.wl_convergence
WANDB_TAGS = args.tags

GPU = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
CPU = torch.device("cpu")
print(f"GPU: {GPU}, CPU: {CPU}")

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
    "Device": GPU,
    "k-fold": K_FOLD,
    "Dataset": DATASET_NAME,
    "learning_rate": LEARNING_RATE,
    "seed": SEED,
    "k_wl": K_WL,
    "model": MODEL_NAME,
    "wl_convergence": WL_CONVERGENCE,
    "tags": WANDB_TAGS
    }
)

wandb.define_metric("epoch")
wandb.define_metric("train accuracy: fold*", step_metric="epoch")
wandb.define_metric("val accuracy: fold*", step_metric="epoch")
wandb.define_metric("train loss: fold*", step_metric="epoch")
wandb.define_metric("val loss: fold*", step_metric="epoch")
wandb.define_metric("train accuracy", summary="last", step_metric="epoch")
wandb.define_metric("val accuracy", summary="last", step_metric="epoch")
wandb.define_metric("train loss", summary="last", step_metric="epoch")
wandb.define_metric("val loss", summary="last", step_metric="epoch")

# Simple training loop
def train(model, loader, optimizer, loss_func):
    # Set model to training mode
    model.train()

    loss_all = 0
    correct = 0
    for data in loader:
        data = data.to(GPU)
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
        data = data.to(GPU)

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
        data = data.to(GPU)
        pred = model(data.x, data.edge_index, data.batch).max(1)[1]
        correct += (pred == data.y).sum().item()
    return correct / len(loader.dataset)

# Set seed for reproducibility
utils.seed_everything(SEED)

# Initialize the model
# First check if the model is a WL model
if MODEL_NAME.startswith("1WL+NN"):
    # Global wl convolution
    wl = WLConv()

    # Load Dataset from https://chrsmrrs.github.io/datasets/docs/datasets/
    dataset = Wrapper_TUDataset(root=f'Code/datasets', name=f'{DATASET_NAME}', use_node_attr=True,
                        pre_transform=WL_Transformer(wl, use_node_attr=True, max_iterations=K_WL, check_convergence=WL_CONVERGENCE), pre_shuffle=True)

    largest_color = len(wl.hashmap)

    if MODEL_NAME == "1WL+NN:Embedding-Sum":
        model = torch_geometric.nn.Sequential('x, edge_index, batch', [
                        (torch.nn.Embedding(num_embeddings=largest_color, embedding_dim=10), 'x -> x'),
                        (torch.squeeze, 'x -> x'),
                        (torch_geometric.nn.pool.global_add_pool, 'x, batch -> x'),
                        (MLP(channel_list=[10, 64, 32, 16, dataset.num_classes]), 'x -> x'),
                        (torch.nn.Softmax(dim=1), 'x -> x')
                    ]).to(GPU)
        
    elif MODEL_NAME == "1WL+NN:Embedding-Max":
        model = torch_geometric.nn.Sequential('x, edge_index, batch', [
                        (torch.nn.Embedding(num_embeddings=largest_color, embedding_dim=10), 'x -> x'),
                        (torch.squeeze, 'x -> x'),
                        (torch_geometric.nn.pool.global_max_pool, 'x, batch -> x'),
                        (MLP(channel_list=[10, 64, 32, 16, dataset.num_classes]), 'x -> x'),
                        (torch.nn.Softmax(dim=1), 'x -> x')
                    ]).to(GPU)
        
elif MODEL_NAME == "GIN:Zero-Sum":
    # Load Dataset from https://chrsmrrs.github.io/datasets/docs/datasets/
    dataset = Wrapper_TUDataset(root=f'tmp2/datasets/{DATASET_NAME}', name=f'{DATASET_NAME}', use_node_attr=True,
                        pre_transform=Constant_Long(0), pre_shuffle=True)

    gin = GIN(in_channels=dataset.num_features, hidden_channels=32, num_layers=5, dropout=0.05, norm='batch_norm', act='relu', jk='cat').to(GPU)
    delattr(gin, 'lin') # Remove the last linear layer that would otherwise remove all jk information
    
    model = torch_geometric.nn.Sequential('x, edge_index, batch', [
                    (gin, 'x, edge_index -> x'),
                    (torch_geometric.nn.pool.global_add_pool, 'x, batch -> x'),
                    (MLP(channel_list=[gin.out_channels * gin.num_layers, 64, 32, 16, dataset.num_classes]), 'x -> x'),
                    (torch.nn.Softmax(dim=1), 'x -> x')
                ]).to(GPU)

else:
    raise ValueError("Invalid model name")

# Log the model to wandb
wandb.watch(model, log="all")

# Split dataset into K_FOLD folds
cross_validation = StratifiedKFold(n_splits=K_FOLD, shuffle=True, random_state=SEED)

# Local logging variables
mean_train_acc = torch.zeros(EPOCHS)
mean_val_acc = torch.zeros(EPOCHS)
mean_train_loss = torch.zeros(EPOCHS)
mean_val_loss = torch.zeros(EPOCHS)

# TRAINING LOOP: Loop over the K_FOLD splits
for fold, (train_ids, test_ids) in enumerate(cross_validation.split(dataset, dataset.y)):
    print(f'Cross-Validation Split {fold+1}/{K_FOLD}:')

    # Reset the model parameters
    model.reset_parameters()

    # Initialize the optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_func = lambda pred, true: torch.nn.CrossEntropyLoss()(pred, true).log().to(GPU)

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
        
        # Log the results locally
        mean_train_acc[epoch] += train_acc
        mean_val_acc[epoch] += val_acc
        mean_train_loss[epoch] += train_loss
        mean_val_loss[epoch] += val_loss

        # Print current status
        if (epoch + 1) % LOG_INTERVAL == 0:
            print(f'\tEpoch: {epoch+1},\t Train Loss: {round(train_loss, 5)},' \
                    f'\t Train Acc: {round(train_acc, 1)}%,\t Val Loss: {round(val_loss, 5)},' \
                    f'\t Val Acc: {round(val_acc, 1)}%')

    
# Averaging the local logging variables
mean_train_acc /= K_FOLD
mean_val_acc /= K_FOLD
mean_train_loss /= K_FOLD
mean_val_loss /= K_FOLD

for epoch in range(EPOCHS):
    wandb.log({"train accuracy": mean_train_acc[epoch], "val accuracy": mean_val_acc[epoch], "train loss": mean_train_loss[epoch], "val loss": mean_val_loss[epoch]})

wandb.finish()
