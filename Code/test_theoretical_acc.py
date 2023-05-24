import torch
from torch_geometric.datasets import TUDataset
from utils import WL_Transformer, Wrapper_TUDataset, seed_everything

import wandb

WL_CONVERGENCE = [True, False]
WL_MAX_ITERATIONS = [-1,1,2,3,4,5,6,7,8]
SEED = 42
DATASET_NAME = "PROTEINS"


run = wandb.init(
    project="BachelorThesis",
    name=f"Accurary {DATASET_NAME}",
    tags=["1WL+NN Theoretical Accuracy"],
    config={
        "dataset_name": DATASET_NAME,
        "seed": SEED,
        "wl_convergence": WL_CONVERGENCE,
        "wl_max_iterations": WL_MAX_ITERATIONS,
    },
)

data_dict = {}
for convergence in WL_CONVERGENCE:

    data_dict[convergence] = {}
    for k_wl in WL_MAX_ITERATIONS:
        
        seed_everything(SEED)

        transformer = WL_Transformer(
            use_node_attr=True,
            max_iterations=k_wl,
            check_convergence=convergence,
        )

        wl_conv = transformer.wl_conv

        dataset = Wrapper_TUDataset(root=f'Code/datasets', name=DATASET_NAME, use_node_attr=False, pre_shuffle=True, pre_transform=[transformer], re_process=True)

        all_color_histograms = [wl_conv.histogram(dataset.get(i).x.squeeze()).squeeze() for i in range(dataset.len())]
        histogram_tensor = torch.stack(all_color_histograms, dim=0)
        unique_values = torch.unique(histogram_tensor, dim=0, sorted=False).shape[0]
         
        data_dict[convergence][k_wl] = f"{round(100 * (unique_values / dataset.len()), 3)}%"
        print(f"Convergence: {convergence}, k_wl: {k_wl}, unique_values: {unique_values}, percentage: {data_dict[convergence][k_wl]}")

table = wandb.Table(columns=['Use WL Convergence'] + [str(i) for i in WL_MAX_ITERATIONS],
            data = [[convergence] + [data_dict[convergence][k_wl] for k_wl in WL_MAX_ITERATIONS] for convergence in WL_CONVERGENCE])
wandb.log({"table_key" : table})