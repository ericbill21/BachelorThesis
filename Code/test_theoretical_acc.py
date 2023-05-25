import torch
from sklearn.decomposition import PCA
from torch_geometric.datasets import TUDataset
from utils import WL_Transformer, Wrapper_TUDataset, seed_everything

import wandb

WL_CONVERGENCE = [True, False]
WL_MAX_ITERATIONS = [-1,1,2,3,4,5,6,7,8]
SEED = 42
DATASET_NAME = "PROTEINS"
MODEL_NAME = "1WL+NN"

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

# We loop over all combinations of convergence and max_iterations
data_dict = {}
for convergence in WL_CONVERGENCE:

    data_dict[convergence] = {}
    for k_wl in WL_MAX_ITERATIONS:
        
        seed_everything(SEED)

        # Load dataset
        transformer = WL_Transformer(
            use_node_attr=True,
            max_iterations=k_wl,
            check_convergence=convergence,
        )
        wl_conv = transformer.wl_conv

        dataset = Wrapper_TUDataset(root=f'Code/datasets', name=DATASET_NAME, use_node_attr=False, pre_shuffle=True, pre_transform=[transformer], reprocess=True)

        # Count unique representations
        all_color_histograms = [wl_conv.histogram(dataset.get(i).x.squeeze()).squeeze() for i in range(dataset.len())]
        histogram_tensor = torch.stack(all_color_histograms, dim=0)
        unique_values = torch.unique(histogram_tensor, dim=0, sorted=False).shape[0]
        
        # Save data and print
        data_dict[convergence][k_wl] = unique_values
        print(f"Convergence: {convergence}, k_wl: {k_wl}, unique_values: {unique_values}, percentage: {(unique_values / dataset.len()) * 100}")


# Log data to wandb
data_total = [[convergence] + [data_dict[convergence][k_wl] for k_wl in WL_MAX_ITERATIONS] for convergence in WL_CONVERGENCE]
data_percentage = [[convergence] + [round((data_dict[convergence][k_wl] / dataset.len()) * 100, 2) for k_wl in WL_MAX_ITERATIONS] for convergence in WL_CONVERGENCE]

data = []
for i in range(dataset.num_classes):
    data.append(data_percentage[i])
    data.append(data_total[i])

table = wandb.Table(columns=['Use WL Convergence'] + [str(i) for i in WL_MAX_ITERATIONS],
            data=data)
wandb.log({"table_key" : table})


