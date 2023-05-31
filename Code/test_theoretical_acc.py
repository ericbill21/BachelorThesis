import torch
from sklearn.decomposition import PCA
from torch_geometric.datasets import TUDataset
from utils import WL_Transformer, Wrapper_WL_TUDataset, seed_everything

import wandb

WL_CONVERGENCE = [False]
WL_MAX_ITERATIONS = [0,1,2,3,4,5,6,7,8]
SEED = 42
DATASET_NAME = "PROTEINS"
MODEL_NAME = "1WL+NN"

run = wandb.init(
    project="BachelorThesisExperiments",
    name=f"Accurary {DATASET_NAME}",
    tags=["1WL+NN Theoretical Accuracy"],
    config={
        "dataset_name": DATASET_NAME,
        "seed": SEED,
        "wl_convergence": WL_CONVERGENCE,
        "wl_max_iterations": WL_MAX_ITERATIONS,
    },
)

global_dataset = TUDataset(root=f"Code/datasets", name=DATASET_NAME, use_node_attr=False)

# We loop over all combinations of convergence and max_iterations
data_dict = {}
for convergence in WL_CONVERGENCE:

    data_dict[convergence] = {}
    for k_wl in WL_MAX_ITERATIONS:
        
        seed_everything(SEED)

        dataset = Wrapper_WL_TUDataset(global_dataset, k_wl=k_wl, wl_convergence=convergence)

        class_indices = {i : [] for i in range(dataset.num_classes)}
        for i in range(dataset.len()):
            class_indices[dataset[i].y.item()].append(i)
        
        # Count unique representations
        class_unique_values = {}
        class_total_values = {}
        class_all_histo = {}
        for i in range(dataset.num_classes):
            # Convert every graph to a color histogram
            if k_wl == 0:
                data_list = [data.x.argmax(dim=-1) for data in dataset[class_indices[i]]]
                max_value = max([data_list[i].max().item() for i in range(len(data_list))])
                all_color_histograms = [torch.tensor([(data_list[i] == j).count_nonzero().item() for j in range(max_value+1)]) for i in range(len(data_list))]
            else:
                all_color_histograms = [dataset.wl_conv.histogram(data.x) for data in dataset[class_indices[i]]]
            class_total_values[i] = len(all_color_histograms)
            class_all_histo[i] = all_color_histograms

            # Count unique color histograms
            histogram_tensor = torch.stack(all_color_histograms, dim=0).squeeze()
            class_unique_values[i] = torch.unique(histogram_tensor, dim=0, sorted=False).shape[0]
        
        # Calc the number of total intersections
        count_intersections = 0
        for i in range(dataset.num_classes):
            for j in range(i+1, dataset.num_classes):
                for histo_i in class_all_histo[i]:
                    for histo_j in class_all_histo[j]:
                        if (histo_i == histo_j).all():
                            count_intersections += 1

        summary = {f"class_{i}" : {
                        "unique_values" : class_unique_values[i],
                        "total_values" : class_total_values[i],
                        "max_accuracy" : class_unique_values[i] / class_total_values[i]
                        } for i in range(dataset.num_classes)} | {
                    "total_unique_values" : dataset.len() - count_intersections,
                    "total_values" : dataset.len(),
                    "max_accuracy" : (dataset.len() - count_intersections) / dataset.len()}
        print(f"Convergence: {convergence}, k_wl: {k_wl}: max_accuracy: {summary['max_accuracy']}")

        wandb.log({f"summary_{convergence}_{k_wl}" : summary})
        data_dict[convergence][k_wl] = summary
    




