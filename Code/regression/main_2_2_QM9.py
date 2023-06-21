from __future__ import division

import argparse
import ast
import sys
import time

import preprocessing as pre
import qm9_datasets as dp

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '.')

import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from models import create_model
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.data import Data, DataLoader, InMemoryDataset
from torch_geometric.nn import GINConv, Set2Set
from utils import Wrapper_WL_TUDataset

import wandb


class QM9(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(QM9, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "QM9_2tew"


    @property
    def processed_file_names(self):
        return "QM9_2etw"

    def download(self):
        pass

    def process(self):
        data_list = []
        targets = dp.get_dataset("QM9", multigregression=True).tolist()
        attributes = pre.get_all_attributes_2_2("QM9")

        node_labels = pre.get_all_node_labels_2_2("QM9", False, False)
        matrices = pre.get_all_matrices_2_2("QM9", list(range(129433)))

        for i, m in enumerate(matrices):
            edge_index_1 = torch.tensor(matrices[i][0]).t().contiguous()
            edge_index_2 = torch.tensor(matrices[i][1]).t().contiguous()

            data = Data()
            data.edge_index_1 = edge_index_1
            data.edge_index_2 = edge_index_2

            one_hot = np.eye(3)[node_labels[i]]
            data.x = torch.from_numpy(one_hot).to(torch.float)

            # Continuous information.
            data.first = torch.from_numpy(np.array(attributes[i][0])[:,0:13]).to(torch.float)
            data.first_coord = torch.from_numpy(np.array(attributes[i][0])[:, 13:]).to(torch.float)

            data.second = torch.from_numpy(np.array(attributes[i][1])[:,0:13]).to(torch.float)
            data.second_coord = torch.from_numpy(np.array(attributes[i][1])[:, 13:]).to(torch.float)
            data.dist = torch.norm(data.first_coord - data.second_coord, p=2, dim=-1).view(-1, 1)
            data.edge_attr = torch.from_numpy(np.array(attributes[i][2])).to(torch.float)
            data.y = torch.from_numpy(np.array([targets[i]])).to(torch.float)

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class MyData(Data):
    def __inc__(self, key, value):
        return self.num_nodes if key in [
            'edge_index_1', 'edge_index_2'
        ] else 0


class MyTransform(object):
    def __call__(self, data):
        new_data = MyData()
        for key, item in data:
            new_data[key] = item
        return new_data


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
            config={
                "dataset": "QM9",
                "k_wl": args.k_wl,
                "model": args.model,
                "wl_convergence": args.wl_convergence} | args.encoding_kwargs | args.mlp_kwargs
)

results = []
results_log = []
for _ in range(5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'QM9')
    dataset = QM9(path, transform=MyTransform()).shuffle()
    dataset.data.y = dataset.data.y[:,0:12]

    def inverse_permutation(perm):
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.size(0), device=perm.device)
        return inv

    # Shuffle the dataset again before calculating the 1-WL colors, and afterward unshuffle it.
    dataset, perm = dataset.shuffle(return_perm=True)
    dataset =  Wrapper_WL_TUDataset(dataset, args.k_wl, args.wl_convergence, DEVICE=device)
    dataset = dataset[inverse_permutation(perm)]

    mean = dataset.data.y.mean(dim=0, keepdim=True)
    std = dataset.data.y.std(dim=0, keepdim=True)
    dataset.data.y = (dataset.data.y - mean) / std
    mean, std = mean.to(device), std.to(device)

    tenpercent = int(len(dataset) * 0.1)
    test_dataset = dataset[:tenpercent].shuffle()
    val_dataset = dataset[tenpercent:2 * tenpercent].shuffle()
    train_dataset = dataset[2 * tenpercent:].shuffle()

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = model = create_model(model_name=args.model,
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

    test_error = None
    test_error_log = None
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
              'Test MAE: {:.7f}, Test MAE: {:.7f}'.format(epoch, lr, loss, val_error, test_error, test_error_log))

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
