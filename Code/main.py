import time
import numpy as np
import pandas as pd
import utils

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


# GLOBAL PARAMETERS
EPOCHS = 100
BATCH_SIZE = 32
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
LOG_INTERVAL = 25
K_FOLD = 10
DATASET_NAME = 'IMDB-MULTI' #'IMDB-MULTI' # 'MUTAG' # 'PROTEINS' # 'IMDB-BINARY' # 'IMDB-MULTI' # 'NCI1' # 'NCI109' # 'DD' # 'COLLAB' # 'ENZYMES' # 'REDDIT-BINARY' # 'REDDIT-MULTI-5K' # 'REDDIT-MULTI-12K' # 'PTC_MR' # 'COX2' # 'DHFR'
PLOT_RESULTS = True
NUM_EPOCHS_TO_BE_PRINTED = 5
SEED = 42

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
    # Set seed for reproducibility
    utils.seed_everything(SEED)

    # Global wl convolution
    wl = WLConv()

    # Dataset from https://chrsmrrs.github.io/datasets/docs/datasets/
    dataset = TUDataset(root=f'Code/datasets/{DATASET_NAME}', name=f'{DATASET_NAME}', 
                        pre_transform=ToDevice(DEVICE))
    dataset = dataset.shuffle()
    
    # Initialize dataset transformer
    wl_transformer = WL_Transformer(wl, use_node_attr=True)
    zero_transformer = Constant_Long(0)

    max_degree = max([data.num_nodes for data in dataset])
    one_hot_degree_transformer = OneHotDegree(max_degree=max_degree)

    # Ugly hack such that the 1-wl algorithm have seen all graphs
    dataset.transform = wl_transformer
    for data in dataset:
        pass

    # Split dataset into K_FOLD folds
    splits  = list(StratifiedKFold(n_splits=K_FOLD, shuffle=True, random_state=SEED).split(dataset, dataset.y))

    # Initialize all models to be tested
    list_of_models = {}

    # Initialize the 1WL+NN models
    # Important: for the 1WL+NN models, the dataset transformer must be set to the wl_transformer!
    total_number_of_colors = len(wl.hashmap)
    
    # 1WL+NN model with Embedding and Summation as its encoding function
    dataset.transform = wl_transformer
    wlnn_model_sum = torch_geometric.nn.Sequential('x, edge_index, batch', [
                    (TorchNN.Embedding(num_embeddings=total_number_of_colors, embedding_dim=10), 'x -> x'),
                    (torch.squeeze, 'x -> x'),
                    (PyGPool.global_add_pool, 'x, batch -> x'),
                    (MLP(channel_list=[10, 60, 40, 20, dataset.num_classes]), 'x -> x'),
                    (TorchNN.Softmax(dim=1), 'x -> x')
                ]).to(DEVICE)
    wlnn_model_sum.dataset_transformer = dataset.transform
    list_of_models['1WL+NN: sum & embedding'] = wlnn_model_sum

    # 1WL+NN model with Embedding and Max as its encoding function
    dataset.transform = wl_transformer
    wlnn_model_max = torch_geometric.nn.Sequential('x, edge_index, batch', [
                    (TorchNN.Embedding(num_embeddings=total_number_of_colors, embedding_dim=10), 'x -> x'),
                    (torch.squeeze, 'x -> x'),
                    (PyGPool.global_max_pool, 'x, batch -> x'),
                    (MLP(channel_list=[10, 60, 40, 20, dataset.num_classes]), 'x -> x'),
                    (TorchNN.Softmax(dim=1), 'x -> x')
                ]).to(DEVICE)
    wlnn_model_max.dataset_transformer = dataset.transform
    list_of_models['1WL+NN: max & embedding'] = wlnn_model_max

    # 1WL+NN model with Embedding and Mean as its encoding function
    dataset.transform = wl_transformer
    wlnn_model_mean = torch_geometric.nn.Sequential('x, edge_index, batch', [
                    (TorchNN.Embedding(num_embeddings=total_number_of_colors, embedding_dim=10), 'x -> x'),
                    (torch.squeeze, 'x -> x'),
                    (PyGPool.global_mean_pool, 'x, batch -> x'),
                    (MLP(channel_list=[10, 60, 40, 20, dataset.num_classes]), 'x -> x'),
                    (TorchNN.Softmax(dim=1), 'x -> x')
                ]).to(DEVICE)
    wlnn_model_mean.dataset_transformer = dataset.transform
    list_of_models['1WL+NN: mean & embedding'] = wlnn_model_mean

    # 1WL+NN model with Embedding and Summation as its encoding function
    dataset.transform = wl_transformer
    wlnn_model_sum = torch_geometric.nn.Sequential('x, edge_index, batch', [
                    (PyGPool.global_add_pool, 'x, batch -> x'),
                    (torch.Tensor.float, 'x -> x'),
                    (MLP(channel_list=[1, 60, 40, 20, dataset.num_classes]), 'x -> x'),
                    (TorchNN.Softmax(dim=1), 'x -> x')
                ]).to(DEVICE)
    wlnn_model_sum.dataset_transformer = dataset.transform
    list_of_models['1WL+NN: sum'] = wlnn_model_sum

    # 1WL+NN model with Embedding and Max as its encoding function
    dataset.transform = wl_transformer
    wlnn_model_max = torch_geometric.nn.Sequential('x, edge_index, batch', [
                    (PyGPool.global_max_pool, 'x, batch -> x'),
                    (torch.Tensor.float, 'x -> x'),
                    (MLP(channel_list=[1, 60, 40, 20, dataset.num_classes]), 'x -> x'),
                    (TorchNN.Softmax(dim=1), 'x -> x')
                ]).to(DEVICE)
    wlnn_model_max.dataset_transformer = dataset.transform
    list_of_models['1WL+NN: max'] = wlnn_model_max

    # 1WL+NN model with Embedding and Mean as its encoding function
    dataset.transform = wl_transformer
    wlnn_model_mean = torch_geometric.nn.Sequential('x, edge_index, batch', [
                    (PyGPool.global_mean_pool, 'x, batch -> x'),
                    (torch.Tensor.float, 'x -> x'),
                    (MLP(channel_list=[1, 60, 40, 20, dataset.num_classes]), 'x -> x'),
                    (TorchNN.Softmax(dim=1), 'x -> x')
                ]).to(DEVICE)
    wlnn_model_mean.dataset_transformer = dataset.transform
    list_of_models['1WL+NN: mean'] = wlnn_model_mean

    # # 1WL+NN model with Embedding and set2set as its encoding function
    # dataset.transform = wl_transformer
    # wlnn_model_mean = torch_geometric.nn.Sequential('x, edge_index, batch', [
    #                 (TorchNN.Embedding(num_embeddings=total_number_of_colors, embedding_dim=10), 'x -> x'),
    #                 (torch.squeeze, 'x -> x'),
    #                 (PyGAggr.Set2Set(in_channels=10, processing_steps=3), 'x, batch -> x'),
    #                 (MLP(channel_list=[10*2, 60, 40, 20, dataset.num_classes]), 'x -> x'),
    #                 (TorchNN.Softmax(dim=1), 'x -> x')
    #             ]).to(DEVICE)
    # wlnn_model_mean.dataset_transformer = dataset.transform
    # list_of_models['1WL+NN: set2set'] = wlnn_model_mean


    # Initialize the GNN models
    # Note that data transformer can be set to anything

    # # GNN model using the GIN construction with the transformer 'zero_transformer'
    # dataset.transform = zero_transformer
    # gin = GIN(in_channels=dataset.num_features, hidden_channels=32, num_layers=5, dropout=0.05, norm='batch_norm', act='relu', jk='cat').to(DEVICE)
    # delattr(gin, 'lin') # Remove the last linear layer that would otherwise remove all jk information
    
    # gnn_model_gin_zero = torch_geometric.nn.Sequential('x, edge_index, batch', [
    #                 (gin, 'x, edge_index -> x'),
    #                 (PyGPool.global_add_pool, 'x, batch -> x'),
    #                 (MLP(channel_list=[gin.out_channels * gin.num_layers, 60, 40, 20, dataset.num_classes]), 'x -> x'),
    #                 (TorchNN.Softmax(dim=1), 'x -> x')
    #             ])
    # gnn_model_gin_zero.dataset_transformer = dataset.transform
    #list_of_models['GIN: sum & zero_transformer'] = gnn_model_gin_zero

    # # GNN model using the GIN construction with the transformer 'one_hot_degree_transformer'
    # dataset.transform = one_hot_degree_transformer
    # gin = GIN(in_channels=dataset.num_features, hidden_channels=32, num_layers=5, dropout=0.05, norm='batch_norm', act='relu', jk='cat').to(DEVICE)
    # delattr(gin, 'lin') # Remove the last linear layer that would otherwise remove all jk information
    
    # gnn_model_gin_degree = torch_geometric.nn.Sequential('x, edge_index, batch', [
    #                 (gin, 'x, edge_index -> x'),
    #                 (PyGPool.global_add_pool, 'x, batch -> x'),
    #                 (MLP(channel_list=[gin.out_channels * gin.num_layers, 60, 40, 20, dataset.num_classes]), 'x -> x'),
    #                 (TorchNN.Softmax(dim=1), 'x -> x')
    #             ])
    # gnn_model_gin_degree.dataset_transformer = dataset.transform
    #list_of_models['GIN: sum & one_hot_degree'] = gnn_model_gin_degree

    # GNN model using the GIN construction with no transformer
    # dataset.transform = None
    # gin = GIN(in_channels=dataset.num_features, hidden_channels=32, num_layers=5, dropout=0.05, norm='batch_norm', act='relu', jk='cat').to(DEVICE)
    # delattr(gin, 'lin') # Remove the last linear layer that would otherwise remove all jk information
    
    # gnn_model_gin_degree = torch_geometric.nn.Sequential('x, edge_index, batch', [
    #                 (gin, 'x, edge_index -> x'),
    #                 (PyGPool.global_add_pool, 'x, batch -> x'),
    #                 (MLP(channel_list=[gin.out_channels * gin.num_layers, 60, 40, 20, dataset.num_classes]), 'x -> x'),
    #                 (TorchNN.Softmax(dim=1), 'x -> x')
    #             ])
    # gnn_model_gin_degree.dataset_transformer = dataset.transform
    # list_of_models['GIN: sum'] = gnn_model_gin_degree


    # Initialize the lists for storing the results
    all_train_losses = {}
    all_train_accuraies = {}
    all_val_losses = {}
    all_val_accuracies = {}

    # TRAINING LOOP
    for model_name, model in list_of_models.items():
        print(f'#'*100 + f'\nTraining: {model_name}')

        # Setting the data transformer
        dataset.transform = model.dataset_transformer

        # Initialize the optimizer and loss function
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=5e-4)
        loss_func = lambda pred, true: TorchNN.CrossEntropyLoss()(pred, true).log()

        # Initialize the lists for storing the results
        train_losses = [[] for _ in range(EPOCHS)]
        train_accuracies = [[] for _ in range(EPOCHS)]
        val_losses = [[] for _ in range(EPOCHS)]
        val_accuracies = [[] for _ in range(EPOCHS)]

        # Training Loop
        runtime = []

        # Loop over the K_FOLD splits
        for i_split, split in enumerate(splits):
            print(f'Cross-Validation Split {i_split+1}/{K_FOLD}:')

            # Reset the model parameters
            model.reset_parameters()

            # Initialize the data loaders
            train_loader = PyGDataLoader(dataset[split[0]], batch_size=BATCH_SIZE, shuffle=True)
            val_loader = PyGDataLoader(dataset[split[1]], batch_size=BATCH_SIZE, shuffle=False)

            # Train the model
            for epoch in range(EPOCHS):
                start = time.time()

                # Train, validate and test the model
                train_loss, train_acc = train(model, train_loader, optimizer, loss_func)
                val_loss, val_acc = val(model, val_loader, loss_func)

                # Save the results
                train_losses[epoch].append(train_loss)
                train_accuracies[epoch].append(train_acc)
                val_losses[epoch].append(val_loss)
                val_accuracies[epoch].append(val_acc)

                # Print current status
                if (epoch + 1) % LOG_INTERVAL == 0:
                    print(f'\tEpoch: {epoch+1},\t Train Loss: {round(train_loss, 5)},' \
                          f'\t Train Acc: {round(train_acc, 1)}%,\t Val Loss: {round(val_loss, 5)},' \
                          f'\t Val Acc: {round(val_acc, 1)}%')
                
                runtime.append(time.time()-start)
            
        print(f'Avg runtime per epoch: {np.mean(runtime)}')

        # Save the results
        all_train_losses[model_name] = train_losses
        all_train_accuraies[model_name] = train_accuracies
        all_val_losses[model_name] = val_losses
        all_val_accuracies[model_name] = val_accuracies
    
    # POST PROCESSING

    # Transform the data into tensors
    for model_name in list_of_models.keys():
        all_train_losses[model_name] = torch.tensor(all_train_losses[model_name])
        all_train_accuraies[model_name] = torch.tensor(all_train_accuraies[model_name])
        all_val_losses[model_name] = torch.tensor(all_val_losses[model_name])
        all_val_accuracies[model_name] = torch.tensor(all_val_accuracies[model_name])

    # Plot the results
    if PLOT_RESULTS:
        visualization.plot_loss_and_accuracy(DATASET_NAME, all_train_losses, all_train_accuraies, all_val_losses, all_val_accuracies)

    # Printing the final results
    epochs = [0] + [i-1 for i in range(EPOCHS // NUM_EPOCHS_TO_BE_PRINTED, EPOCHS + EPOCHS // NUM_EPOCHS_TO_BE_PRINTED, EPOCHS // NUM_EPOCHS_TO_BE_PRINTED)]

    print(f'\nFinal validation accuracies:')
    print(f'{"Epoch:" : <30} {"".join([f"{e + 1 : ^15}" for e in epochs])}')
    for model_name in list_of_models.keys():
        res = ''.join([f"{f'{round(all_val_accuracies[model_name][e].mean().item(), 1)}±{round(all_val_accuracies[model_name][e].std().item(), 1)}%' : ^15}" for e in epochs])
        print(f'{model_name : <30} {res}')

if __name__ == '__main__':
    main()