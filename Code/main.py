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
from wlnn import constant_and_id_transformer

from torch_geometric.nn.conv import wl_conv

from torch_geometric.transforms.constant import Constant

from torch_geometric.loader import DataLoader

def main():

    # Dataset from https://chrsmrrs.github.io/datasets/docs/datasets/
    dataset = TUDataset(root='/tmp/IMDB-MULTI', name='IMDB-MULTI', transform=constant_and_id_transformer)
    dataset = dataset.shuffle()

    # Split dataset into training and test set
    fraction  = 0.9
    split = int(fraction * dataset.len())
    train_dataset = dataset[:split]
    test_dataset = dataset[split:]

    wlnn_model = WLNN()
    wlnn_model.init_training(train_dataset, test_dataset)

    total_colors = wlnn_model.get_total_number_of_colors()

    # Initialize and set a simple MLP
    mlp = nn.Sequential(
            nn.Linear(total_colors, 60),
            nn.ReLU(),
            nn.Linear(60, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.Softmax(),
            nn.Linear(20, train_dataset.num_classes))
    
    wlnn_model.set_mlp(mlp)

    # Initialize and set the counting encoding function
    f_enc = create_counting_encoding(total_colors)
    wlnn_model.set_encoding(f_enc)

    # Train the model
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(20):
        train_dataset_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        for i in range(10):
            batch = next(iter(train_dataset_loader))

            optimizer.zero_grad()
            out = wlnn_model.forward(batch.x, batch.edge_index, batch.num_graphs, batch.ptr)

            # We need to convert the labels to one-hot encoding of the probabilities we expect
            # Example: graph.y = [2] -> y = torch.tensor([0.0, 0.0, 1.0])
            y = nn.functional.one_hot(batch.y, train_dataset.num_classes).float().squeeze(0)

            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

            print(f'Epoch:\t {epoch}, Batch:\t {i}, Loss:\t {loss}')
        
    print('Hey')


def create_counting_encoding(n):

    def counting_encoding(x):
        out = torch.zeros(n)
        for i in range(n):
            out[i] = torch.sum(x == i)
        
        return out

    return counting_encoding
    



if __name__ == '__main__':
    main()