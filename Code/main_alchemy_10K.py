import sys

sys.path.insert(0, '..')
sys.path.insert(0, '.')

import argparse
import ast
import os.path as osp
import time

import numpy as np
import torch
import torch.nn.functional as F
from models import create_model
from torch.nn import Linear
from torch.nn import Linear as Lin
from torch.nn import ReLU, Sequential
from torch_geometric.data import Data, DataLoader, InMemoryDataset
from torch_geometric.datasets import TUDataset
from utils import Wrapper_WL_TUDataset

import wandb

parser = argparse.ArgumentParser(description='BachelorThesisExperiments')
parser.add_argument('--k_wl', type=int, help='Number of Weisfeiler-Lehman iterations, or if -1 it runs until convergences.')
parser.add_argument('--model', type=str, default='1WL+NN:Embedding-Sum', help='Model to use. Options are "1WL+NN:Embedding-{SUM,MAX,MEAN}" or "GIN:{SUM,MAX,MEAN}".')
parser.add_argument('--wl_convergence', type=str, choices=['True','False'], help='Whether to use the convergence criterion for the Weisfeiler-Lehman algorithm.')
parser.add_argument('--transformer_kwargs', type=str, default='{}', help='Arguments for the transformer. For example, for the OneHotDegree transformer, the argument is the maximum degree.')
parser.add_argument('--encoding_kwargs', type=str, default='{}', help='Arguments for the encoding function. For example, for Embedding, the argument is the embedding dimension with the key "embedding_dim".')
parser.add_argument('--mlp_kwargs', type=str, default='{}', help='Arguments for the MLP. For example, for the MLP, the argument is the number of hidden layers with the key "num_layers".')
parser.add_argument('--tags', nargs='+', default=[], help='Tags for the run on wandb.')
args = parser.parse_args()

args.wl_convergence = args.wl_convergence == "True"
args.encoding_kwargs = ast.literal_eval(args.encoding_kwargs)
args.mlp_kwargs = ast.literal_eval(args.mlp_kwargs)

wandb.init(project="BachelorThesisExperiments",
            name=f"{args.model}: {time.strftime('%d.%m.%Y %H:%M:%S')}",
            tags=args.tags,
            config={
                "dataset": "Alchemy10K",
                "k_wl": args.k_wl,
                "model": args.model,
                "wl_convergence": args.wl_convergence} | args.encoding_kwargs | args.mlp_kwargs
)

plot_all = []
results = []
results_log = []
for _ in range(5):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    plot_it = []
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'datasets', "alchemy_full")

    infile = open("train_al_10.index", "r")
    for line in infile:
        indices_train = line.split(",")
        indices_train = [int(i) for i in indices_train]

    infile = open("val_al_10.index", "r")
    for line in infile:
        indices_val = line.split(",")
        indices_val = [int(i) for i in indices_val]

    infile = open("test_al_10.index", "r")
    for line in infile:
        indices_test = line.split(",")
        indices_test = [int(i) for i in indices_test]

    indices = indices_train
    indices.extend(indices_val)
    indices.extend(indices_test)

    def inverse_permutation(perm):
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.size(0), device=perm.device)
        return inv

    dataset, perm= TUDataset(path, name="alchemy_full")[indices].shuffle(return_perm=True)
    dataset =  Wrapper_WL_TUDataset(dataset, args.k_wl, args.wl_convergence, DEVICE=device)
    dataset = dataset[inverse_permutation(perm)]

    args.encoding_kwargs["max_node_feature"] = dataset.max_node_feature + 1

    print(len(dataset)) 

    mean = dataset.data.y.mean(dim=0, keepdim=True)
    std = dataset.data.y.std(dim=0, keepdim=True)
    dataset.data.y = (dataset.data.y - mean) / std
    mean, std = mean.to(device), std.to(device)

    train_dataset = dataset[0:10000]
    val_dataset = dataset[10000:11000]
    test_dataset = dataset[11000:]

    batch_size = 25
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = create_model(model_name=args.model,
                         input_dim=dataset.num_features,
                         output_dim=12,
                         mlp_kwargs = args.mlp_kwargs,
                         gnn_kwargs = {},
                         encoding_kwargs = args.encoding_kwargs,
                         is_classification=False).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.5, patience=5,
                                                           min_lr=0.0000001)

    def train():
        model.train()
        loss_all = 0

        lf = torch.nn.L1Loss()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss = lf(model(data), data.y)

            loss.backward()
            loss_all += loss.item() * data.num_graphs
            optimizer.step()
        return (loss_all / len(train_loader.dataset))


    @torch.no_grad()
    def test(loader):
        model.eval()
        error = torch.zeros([1, 12]).to(device)

        for data in loader:
            data = data.to(device)
            error += ((data.y * std - model(data) * std).abs() / std).sum(dim=0)

        error = error / len(loader.dataset)
        error_log = torch.log(error)

        return error.mean().item(), error_log.mean().item()


    best_val_error = None
    for epoch in range(1, 1001):
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train()
        val_error, _ = test(val_loader)

        scheduler.step(val_error)
        if best_val_error is None or val_error <= best_val_error:
            test_error, test_error_log = test(test_loader)
            best_val_error = val_error

        print('Epoch: {:03d}, LR: {:.7f}, Loss: {:.7f}, Validation MAE: {:.7f}, '
              'Test MAE: {:.7f},Test MAE: {:.7f}, '.format(epoch, lr, loss, val_error, test_error, test_error_log))

        if lr < 0.000001:
            print("Converged.")
            break


    results.append(test_error)
    results_log.append(test_error_log)

print("########################")
print(results)
results = np.array(results)
print(results.mean(), results.std())

print(results_log)
results_log = np.array(results_log)
print(results_log.mean(), results_log.std())

wandb.log({"test_error": results.mean(), "test_error_std": results.std(), "test_error_log": results_log.mean(), "test_error_log_std": results_log.std()})
wandb.finish()