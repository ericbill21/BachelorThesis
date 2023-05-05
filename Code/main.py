# TODO: Beutify the imports
import torch
import torch.nn as nn
from torch_geometric.datasets import TUDataset
from wlnn import WLNN
from wlnn import create_1wl_transformer
from wlnn import wl_algorithm
from encoding import *
from torch_geometric.nn.conv import wl_conv
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader as TorchDataLoader
import time
import numpy as np
from torch_geometric.nn.models import GIN, basic_gnn
from torch_geometric.nn import global_mean_pool
import torch_geometric

from torch_geometric.nn.conv import GINConv
from torch_geometric.nn.models import MLP

# GLOBAL PARAMETERS
TRAINING_FRACTION = 0.9
EPOCHS = 100
BATCH_SIZE = 32
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DATASET_NAME = 'IMDB-BINARY'

def train(model, loader, optimizer, loss_func):
    model.train()
    loss_all = 0

    for data in loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()

        loss = loss_func(model(data.x, data.edge_index, data.batch), data.y)

        loss.backward()
        optimizer.step()
        loss_all += data.num_graphs * loss.item()
    return loss_all / len(loader.dataset)

def val(model, loader, loss_func):
    model.eval()
    loss_all = 0

    for data in loader:
        data = data.to(DEVICE)
        loss_all += loss_func(model(data.x, data.edge_index, data.batch), data.y).item()

    return loss_all / len(loader.dataset)

def test(model, loader):
    model.eval()
    correct = 0

    for data in loader:
        data = data.to(DEVICE)
        pred = model(data.x, data.edge_index, data.batch).max(1)[1]
        correct += (pred == data.y).sum().item()
    return correct / len(loader.dataset)

def main():
    # Global wl convolution
    wl = wl_conv.WLConv()

    transformer = create_1wl_transformer(wl)

    # Dataset from https://chrsmrrs.github.io/datasets/docs/datasets/
    dataset = TUDataset(root=f'Code/datasets/{DATASET_NAME}', name=f'{DATASET_NAME}', transform=transformer)
    dataset = dataset.shuffle()

    # Ugly hack such that the 1-wl algorithm have seen all graphs
    for data in dataset:
        pass

    # Split dataset into training and test set
    split = int(TRAINING_FRACTION * dataset.len())
    train_dataset = dataset[:split]
    test_dataset = dataset[split:]

    # Initialize the wlnn model
    total_number_of_colors = len(wl.hashmap) + 1 # We add 1 for safety reasons as the number of colors used by the wl algorihtm fluctuates by 1
    
    wlnn_model = torch_geometric.nn.Sequential('x, edge_index, batch', [
                    (nn.Embedding(total_number_of_colors, 10), 'x -> x'),
                    (torch.squeeze, 'x -> x'),
                    (global_mean_pool, 'x, batch -> x'),
                    (MLP([10, 60, 40, 20, dataset.num_classes]), 'x -> x'),
                    (nn.Softmax(dim=1), 'x -> x')
                ])

    # Initialize the GNN model
    gnn_model = torch_geometric.nn.Sequential('x, edge_index, batch', [
                    (GINConv(MLP([dataset.num_features, 5, 10])), 'x, edge_index -> x'),
                    (GINConv(MLP([10, 10, 10])), 'x, edge_index -> x'),
                    (GINConv(MLP([10, 5, 3])), 'x, edge_index -> x'),
                    (global_mean_pool, 'x, batch -> x'),
                    (MLP([3, dataset.num_classes]), 'x -> x'),
                    (nn.Softmax(dim=1), 'x -> x')
                ])

    model = wlnn_model


    # Initialize the optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=5e-4)
    loss_func = nn.functional.nll_loss #nn.CrossEntropyLoss()

    # Initialize the data loaders
    train_loader = PyGDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = PyGDataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Training Loop
    runtime = []
    for epoch in range(1, EPOCHS):
        start = time.time()

        # Train and validate the model
        train_loss = train(model, train_loader, optimizer, loss_func)
        val_loss = val(model, val_loader, loss_func)

        if epoch % 5 == 0:
            # Test the accuracy of the model
            acc = test(model, val_loader)

            print(f'Epoch: {round(epoch, 3)},\t Train Loss: {round(train_loss, 5)},\t Val Loss: {round(val_loss, 5)},\t Val Acc: {round(acc, 2)}')
        
        runtime.append(time.time()-start)
    
    print(f'Avg runtime per epoch: {np.mean(runtime)}')


if __name__ == '__main__':
    main()