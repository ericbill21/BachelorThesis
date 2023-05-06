# TODO: Beutify the imports
import torch
import torch.nn as nn
from torch_geometric.datasets import TUDataset
from wlnn import WLNN
from wlnn import create_1wl_transformer, Constant_Long
from encoding import *
from torch_geometric.nn.conv import wl_conv
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader as TorchDataLoader
import time
import numpy as np
from torch_geometric.nn.models import GIN, basic_gnn
from torch_geometric.nn import pool as PyG_pool
import torch_geometric
from matplotlib import pyplot as plt

from torch_geometric.nn.conv import GINConv
from torch_geometric.nn.models import MLP

# GLOBAL PARAMETERS
TRAINING_FRACTION = 0.8
EPOCHS = 500
BATCH_SIZE = 32
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DATASET_NAME = 'IMDB-BINARY'
LOG_INTERVAL = 50

# Simple training loop
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

# Simple validation loop
def val(model, loader, loss_func):
    model.eval()
    loss_all = 0

    for data in loader:
        data = data.to(DEVICE)
        loss_all += loss_func(model(data.x, data.edge_index, data.batch), data.y).item()

    return loss_all / len(loader.dataset)

# Simple test loop
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
                ])
    list_of_models['1WL+NN: sum'] = wlnn_model_sum

    # 1WL+NN model with Embedding and Max as its encoding function
    wlnn_model_max = torch_geometric.nn.Sequential('x, edge_index, batch', [
                    (nn.Embedding(total_number_of_colors, 10), 'x -> x'),
                    (torch.squeeze, 'x -> x'),
                    (PyG_pool.global_max_pool, 'x, batch -> x'),
                    (MLP([10, 60, 40, 20, dataset.num_classes]), 'x -> x'),
                    (nn.Softmax(dim=1), 'x -> x')
                ])
    list_of_models['1WL+NN: max'] = wlnn_model_max

    # 1WL+NN model with Embedding and Mean as its encoding function
    wlnn_model_mean = torch_geometric.nn.Sequential('x, edge_index, batch', [
                    (nn.Embedding(total_number_of_colors, 10), 'x -> x'),
                    (torch.squeeze, 'x -> x'),
                    (PyG_pool.global_mean_pool, 'x, batch -> x'),
                    (MLP([10, 60, 40, 20, dataset.num_classes]), 'x -> x'),
                    (nn.Softmax(dim=1), 'x -> x')
                ])
    list_of_models['1WL+NN: mean'] = wlnn_model_mean

    # Initialize the GNN model
    gnn_model_gin = torch_geometric.nn.Sequential('x, edge_index, batch', [
                    (GINConv(MLP([dataset.num_features, 5, 10])), 'x, edge_index -> x'),
                    (GINConv(MLP([10, 10, 10])), 'x, edge_index -> x'),
                    (GINConv(MLP([10, 5, 3])), 'x, edge_index -> x'),
                    (PyG_pool.global_mean_pool, 'x, batch -> x'),
                    (MLP([3, dataset.num_classes]), 'x -> x'),
                    (nn.Softmax(dim=1), 'x -> x')
                ])
    list_of_models['GNN: GIN'] = gnn_model_gin

    # gnn_model = torch_geometric.nn.Sequential('x, edge_index, batch', [
    #                 (torch_geometric.nn.models.GIN(dataset.num_features, 10, 5, 10), 'x, edge_index -> x'),
    #                 (PyG_pool.global_mean_pool, 'x, batch -> x'),
    #                 (MLP([10, dataset.num_classes]), 'x -> x'),
    #                 (nn.Softmax(dim=1), 'x -> x')
    #             ])
    # list_of_models['gin2'] = gnn_model

    # Initialize the lists for storing the results
    all_train_losses = {}
    all_val_losses = {}
    all_test_accuracies = {}

    # TRAINING LOOP
    for model_name, model in list_of_models.items():
        print(f'#'*50 + f'\nTraining: {model_name}')

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
        val_losses = []
        test_accuracies = []

        # Training Loop
        runtime = []
        for epoch in range(1, EPOCHS):
            start = time.time()

            # Train, validate and test the model
            train_loss = train(model, train_loader, optimizer, loss_func)
            val_loss = val(model, val_loader, loss_func)
            acc = test(model, val_loader)

            # Save the results
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            test_accuracies.append(acc)

            # Print current status
            if epoch % LOG_INTERVAL == 0:
                print(f'Epoch: {round(epoch, 3)},\t Train Loss: {round(train_loss, 5)},\t Val Loss: {round(val_loss, 5)},\t Val Acc: {round(acc, 2)}')
            
            runtime.append(time.time()-start)
        
        print(f'Avg runtime per epoch: {np.mean(runtime)}')
    
        # Save the results
        all_train_losses[model_name] = train_losses
        all_val_losses[model_name] = val_losses
        all_test_accuracies[model_name] = test_accuracies
    
    # Plot the results
    # TODO: Auslagern in eigene datei und adde ticks, check siemens notebook
    fig, axs = plt.subplots(2, 2)

    axs[0, 0].set_title('Training loss')
    axs[0, 1].set_title('Validation loss')
    axs[1, 0].set_title('Test accuracy')

    axs[0, 0].set_xlabel('Epoch')
    axs[1, 0].set_xlabel('Epoch')
    axs[0, 1].set_xlabel('Epoch')

    axs[0, 0].set_ylabel('Loss')
    axs[0, 1].set_ylabel('Loss')
    axs[1, 0].set_ylabel('Accuracy')

    axs[0, 0].grid(True)
    axs[0, 1].grid(True)
    axs[1, 0].grid(True)

    color_map = {}    
    colors = iter(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    
    for model_name in list_of_models.keys():
        color_map[model_name] = next(colors)

        axs[0, 0].plot(all_train_losses[model_name], label=model_name, c=color_map[model_name])
        axs[0, 1].plot(all_val_losses[model_name], label=model_name, c=color_map[model_name])
        axs[1, 0].plot(all_test_accuracies[model_name], label=model_name, c=color_map[model_name])
    
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.show()


if __name__ == '__main__':
    main()