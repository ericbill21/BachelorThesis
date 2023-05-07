# TODO: Beutify the imports
import torch
import torch.nn as nn
from torch_geometric.datasets import TUDataset
from utils import create_1wl_transformer, Constant_Long
from torch_geometric.nn.conv import wl_conv
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader as TorchDataLoader
import time
import numpy as np
from torch_geometric.nn.models import GIN
from torch_geometric.nn import pool as PyG_pool
import torch_geometric

from torch_geometric.nn.conv import GINConv
from torch_geometric.nn.models import MLP

from visualization import plot_loss

# GLOBAL PARAMETERS
TRAINING_FRACTION = 0.8
EPOCHS = 500
BATCH_SIZE = 32
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DATASET_NAME = 'IMDB-BINARY'
LOG_INTERVAL = 10

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

def main():
    # Global wl convolution
    wl = wl_conv.WLConv()

    wl_transformer = create_1wl_transformer(wl)
    c_transformer = Constant_Long(0)

    # Dataset from https://chrsmrrs.github.io/datasets/docs/datasets/
    dataset = TUDataset(root=f'Code/datasets/{DATASET_NAME}', name=f'{DATASET_NAME}', transform=wl_transformer)
    dataset = dataset.shuffle()

    # Ugly hack such that the 1-wl algorithm have seen all graphs
    for data in dataset:
        pass

    # Split dataset into training and test set
    split = int(TRAINING_FRACTION * dataset.len())
    train_dataset = dataset[:split]
    test_dataset = dataset[split:]

    # Initialize the wlnn models
    total_number_of_colors = len(wl.hashmap)
    
    list_of_models = {}

    # 1WL+NN model with Embedding and Summation as its encoding function
    wlnn_model_sum = torch_geometric.nn.Sequential('x, edge_index, batch', [
                    (nn.Embedding(total_number_of_colors, 10), 'x -> x'),
                    (torch.squeeze, 'x -> x'),
                    (PyG_pool.global_add_pool, 'x, batch -> x'),
                    (MLP([10, 60, 40, 20, dataset.num_classes]), 'x -> x'),
                    (nn.Softmax(dim=1), 'x -> x')
                ]).to(DEVICE)
    list_of_models['1WL+NN: sum'] = wlnn_model_sum

    # 1WL+NN model with Embedding and Max as its encoding function
    wlnn_model_max = torch_geometric.nn.Sequential('x, edge_index, batch', [
                    (nn.Embedding(total_number_of_colors, 10), 'x -> x'),
                    (torch.squeeze, 'x -> x'),
                    (PyG_pool.global_max_pool, 'x, batch -> x'),
                    (MLP([10, 60, 40, 20, dataset.num_classes]), 'x -> x'),
                    (nn.Softmax(dim=1), 'x -> x')
                ]).to(DEVICE)
    #list_of_models['1WL+NN: max'] = wlnn_model_max

    # 1WL+NN model with Embedding and Mean as its encoding function
    wlnn_model_mean = torch_geometric.nn.Sequential('x, edge_index, batch', [
                    (nn.Embedding(total_number_of_colors, 10), 'x -> x'),
                    (torch.squeeze, 'x -> x'),
                    (PyG_pool.global_mean_pool, 'x, batch -> x'),
                    (MLP([10, 60, 40, 20, dataset.num_classes]), 'x -> x'),
                    (nn.Softmax(dim=1), 'x -> x')
                ]).to(DEVICE)
    #list_of_models['1WL+NN: mean'] = wlnn_model_mean

    # Initialize the GNN models
    gin = GIN(dataset.num_features, 32, 5, dropout=0.05, norm='batch_norm', act='relu', jk='cat').to(DEVICE)
    gin.lin = torch.nn.Identity() # Remove the last linear layer that would remove all jk information
    
    gnn_model_gin = torch_geometric.nn.Sequential('x, edge_index, batch', [
                    (gin, 'x, edge_index -> x'),
                    (PyG_pool.global_add_pool, 'x, batch -> x'),
                    (MLP([gin.out_channels * gin.num_layers, 60, 40, 20, dataset.num_classes]), 'x -> x'),
                    (nn.Softmax(dim=1), 'x -> x')
                ])

    list_of_models['GIN'] = gnn_model_gin

    # Initialize the lists for storing the results
    all_train_losses = {}
    all_train_accuraies = {}
    all_val_losses = {}
    all_val_accuracies = {}

    # TRAINING LOOP
    for model_name, model in list_of_models.items():
        print(f'#'*50 + f'\nTraining: {model_name}')

        # TODO: make it better: Maybe check if first entry of model is gnn?
        if model_name == 'gin':
            train_dataset.transform = c_transformer
            test_dataset.transform = c_transformer
        else:
            train_dataset.transform = wl_transformer
            test_dataset.transform = wl_transformer

        # Initialize the optimizer and loss function
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=5e-4)
        CE = nn.CrossEntropyLoss()
        loss_func = lambda pred, true: CE(pred, true).log()

        # Initialize the data loaders
        train_loader = PyGDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = PyGDataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Initialize the lists for storing the results
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        # Training Loop
        runtime = []
        for epoch in range(1, EPOCHS):
            start = time.time()

            # Train, validate and test the model
            train_loss, train_acc = train(model, train_loader, optimizer, loss_func)
            val_loss, val_acc = val(model, val_loader, loss_func)

            # Save the results
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            # Print current status
            if epoch % LOG_INTERVAL == 0:
                print(f'Epoch: {round(epoch, 3)},\t Train Loss: {round(train_loss, 5)},\t Train Acc: {round(train_acc, 1)}%,\t Val Loss: {round(val_loss, 5)},\t Val Acc: {round(val_acc, 1)}%')
            
            runtime.append(time.time()-start)
        
        print(f'Avg runtime per epoch: {np.mean(runtime)}')
    
        # Save the results
        all_train_losses[model_name] = train_losses
        all_train_accuraies[model_name] = train_accuracies
        all_val_losses[model_name] = val_losses
        all_val_accuracies[model_name] = val_accuracies
    
    # Plot the results
    plot_loss(all_train_losses, all_train_accuraies, all_val_losses, all_val_accuracies)


if __name__ == '__main__':
    main()