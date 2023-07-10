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
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GINConv, MessagePassing, global_mean_pool
from utils import Wrapper_WL_TUDataset

import wandb

parser = argparse.ArgumentParser(description='BachelorThesisExperiments')
parser.add_argument('--model', type=str, default='1WL+NN:Embedding-Sum', help='Model to use. Options are "1WL+NN:Embedding-{SUM,MAX,MEAN}" or "GIN:{SUM,MAX,MEAN}".')
parser.add_argument('--gnn_kwargs', type=str, default='{}', help='Arguments for the GNN. For example, for GIN, the argument is the number of MLP layers with the key "num_layers".')
parser.add_argument('--mlp_kwargs', type=str, default='{}', help='Arguments for the MLP. For example, for the MLP, the argument is the number of hidden layers with the key "num_layers".')
parser.add_argument('--tags', nargs='+', default=[], help='Tags for the run on wandb.')
args = parser.parse_args()

args.mlp_kwargs = ast.literal_eval(args.mlp_kwargs)
args.gnn_kwargs = ast.literal_eval(args.gnn_kwargs)

wandb.init(project="BachelorThesisExperiments",
            name=f"{args.model}: {time.strftime('%d.%m.%Y %H:%M:%S')}",
            tags=args.tags,
            config={
                "dataset": "Zinc",
                "model": args.model} | args.gnn_kwargs | args.mlp_kwargs
)


results = []
results_log = []
for _ in range(5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'datasets', 'ZINC_full')

    dataset = TUDataset(path, name="ZINC_full")

    train_dataset = dataset[0:220011].shuffle()
    val_dataset = dataset[225011:249456].shuffle()
    test_dataset = dataset[220011:225011].shuffle()

    print(f"Size of training dataset: {len(train_dataset)}, size of validation dataset: {len(val_dataset)}, size of test dataset: {len(test_dataset)}")

    batch_size = 25
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = create_model(model_name=args.model,
                        input_dim=dataset.num_features,
                        output_dim=1,
                        mlp_kwargs = args.mlp_kwargs,
                        gnn_kwargs = args.gnn_kwargs,
                        encoding_kwargs = {},
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
            loss = lf(model(data).squeeze(), data.y)

            loss.backward()
            loss_all += loss.item() * data.num_graphs
            optimizer.step()
        return (loss_all / len(train_loader.dataset))

    @torch.no_grad()
    def test(loader):
        model.eval()
        error = torch.tensor(0.0).to(device)

        for data in loader:
            data = data.to(device)
            error += (model(data).squeeze() - data.y).abs().sum()

        error = error / len(loader.dataset)
        error_log = torch.log(error)

        return error.item(), error_log.item()


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
              'Test MAE: {:.7f}'.format(epoch, lr, loss, val_error, test_error))

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