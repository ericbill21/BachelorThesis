import wandb
import torch
import utils
from torch_geometric.datasets import TUDataset
from torch_geometric.nn.conv import WLConv

wl_conv = WLConv()

wandb.init(
    # set the wandb project where this run will be logged
    project="BachelorThesis",
    name=f"WL_Iterations",
)

wandb.define_metric("WL_iterations")
wandb.define_metric("WL_Count", steps_metric="WL_iterations")

dataset = TUDataset(root='tmp', name='PROTEINS')

wl_trans = utils.WL_Transformer(wl_conv, use_node_attr=True, max_iterations=-1, check_convergence=True)

res = torch.zeros(dataset.len())
for idx, data in enumerate(dataset):
    res[idx] = wl_trans(data)

res = res.unique(return_counts=True)
for iter, count in zip(res[0], res[1]):
    wandb.log({"WL_iterations": iter, "WL_Count": 100 * count / dataset.len()})
    print(iter, 100* count / dataset.len())
