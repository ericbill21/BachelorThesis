import torch
import torch.nn as nn
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from matplotlib import pyplot as plt
import networkx as nx
from torch_geometric import utils
from torch_geometric.data import Data
from torch_geometric.nn.pool import global_add_pool
from torch_geometric.datasets import TUDataset

from wlnn import WLNN
from wlnn import create_transformer
from wlnn import wl_algorithm

from encoding import *

from torch_geometric.nn.conv import wl_conv

from torch_geometric.transforms.constant import Constant

from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader as TorchDataLoader

from torch.utils.data import Dataset

import time
import numpy as np

from torch.nn import functional

from tabulate import tabulate

TRAINING_FRACTION = 0.9
EPOCHS = 3000
BATCH_SIZE = 32
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def train(model, loader, optimizer, loss_func):
    model.train()
    loss_all = 0

    for data in loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        loss = loss_func(model(data), data.y)
        loss.backward()
        optimizer.step()
        loss_all += data.num_graphs * loss.item()
    return loss_all / len(loader.dataset)


def val(model, loader, loss_func):
    model.eval()
    loss_all = 0

    for data in loader:
        data = data.to(DEVICE)
        loss_all += loss_func(model(data), data.y).item()
    return loss_all / len(loader.dataset)


def test(model, loader):
    model.eval()
    correct = 0

    for data in loader:
        data = data.to(DEVICE)
        pred = model(data).max(1)[1]
        correct += (pred == data.y).sum().item()
    return correct / len(loader.dataset)

def main():
    # Global wl convolution
    wl = wl_conv.WLConv()

    # Dataset from https://chrsmrrs.github.io/datasets/docs/datasets/
    dataset = TUDataset(root='Code/datasets/IMDB-MULTI', name='IMDB-MULTI', transform=create_transformer(wl))
    dataset = dataset.shuffle()

    # Ugly hack such that the 1-wl algorithm have seen all graphs
    for data in dataset:
        pass

    # Split dataset into training and test set
    split = int(TRAINING_FRACTION * dataset.len())
    train_dataset = dataset[:split]
    test_dataset = dataset[split:]

    # Initialize and set the counting encoding function
    total_number_of_colors = len(wl.hashmap) + 1 # We add 1 for safety reasons as the number of colors used by the wl algorihtm fluctuates by 1
    embedding = nn.Embedding(total_number_of_colors, 25)

    f_enc = Mean_Encoding(embedding)

    # Initialize and set a simple MLP
    mlp = nn.Sequential(
            nn.Linear(f_enc.out_dim, 60),
            nn.ReLU(),
            nn.Linear(60, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, train_dataset.num_classes),
            nn.Softmax(dim=1))
    
    # Initialize the WLNN model
    wlnn_model = WLNN(f_enc=f_enc, mlp=mlp)

    # Initialize the optimizer and loss function
    params = list(wlnn_model.parameters()) + list(mlp.parameters()) + list(embedding.parameters())
    optimizer = torch.optim.Adam(params, lr=0.01, weight_decay=5e-4)
    loss_func = nn.CrossEntropyLoss()

    # Initialize the data loaders
    train_loader = PyGDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = PyGDataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Training Loop
    runtime = []
    for epoch in range(1, EPOCHS):
        start = time.time()

        # Train and validate the model
        train_loss = train(wlnn_model, train_loader, optimizer, loss_func)
        val_loss = val(wlnn_model, val_loader, loss_func)

        if epoch % 25 == 0:
            # Test the accuracy of the model
            acc = test(wlnn_model, val_loader)

            print(f'Epoch: {round(epoch, 3)},\t Train Loss: {round(train_loss, 5)},\t Val Loss: {round(val_loss, 5)},\t Val Acc: {round(acc, 2)}')
        
        runtime.append(time.time()-start)
    
    print(f'Avg runtime per epoch: {np.mean(runtime)}')


if __name__ == '__main__':
    main()