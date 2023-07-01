import argparse

import torch
from sklearn.decomposition import PCA
from torch_geometric.datasets import TUDataset
from utils import Constant_Long, WL_Transformer, Wrapper_WL_TUDataset, seed_everything

import wandb


def calculate_max_accuracy(x, y):
        # Combine x and y into a single dataset
        dataset = torch.cat((x, y.unsqueeze(1)), dim=1)

        # Get unique samples in x
        unique_samples, unique_indices = torch.unique(x, dim=0, return_inverse=True)

        max_correct = 0
        total_samples = 0

        for i in range(unique_samples.shape[0]):
            # Find indices of matching samples in the dataset. 
            # Necessary to use flatten() to get a 1D tensor such that 'dataset[matching_indices]' returns a 2D tensor.
            matching_indices = torch.nonzero(unique_indices == i, as_tuple=False).flatten()

            # Get matching samples and labels
            matching_samples = dataset[matching_indices]
            matching_labels = matching_samples[:, -1]

            # Count the occurrences of each class label
            _, label_counts = torch.unique(matching_labels, return_counts=True)

            # Update the maximum correct count
            max_correct += torch.max(label_counts)

            # Update the total number of samples
            total_samples += matching_samples.shape[0]

        # Calculate the maximal accuracy
        max_accuracy = max_correct / total_samples
        return max_accuracy

def main():

    parser = argparse.ArgumentParser(description='BachelorThesisExperiments')
    parser.add_argument('--dataset', type=str, help='Dataset')
    parser.add_argument('--max_iterations', type=int, help='Maximum number of iterations for the Weisfeiler-Lehman algorithm')
    args = parser.parse_args()

    # PARAMETERS
    WL_CONVERGENCE = [False]
    WL_MAX_ITERATIONS = list(range(0, args.max_iterations + 1))
    SEED = 42
 
    # Initialize wandb
    run = wandb.init(
        project="BachelorThesisExperiments",
        name=f"{args.dataset}: Theoretical Accuracy",
        tags=["Accuracy"],
        config={
            "seed": SEED,
            "wl_convergence": WL_CONVERGENCE,
            "wl_max_iterations": WL_MAX_ITERATIONS,
            "dataset": args.dataset
        },
    )

    # Load dataset
    global_dataset = TUDataset(root=f"Code/datasets", name=args.dataset, use_node_attr=False)

    # Create constant node feature if there are no node features.
    if global_dataset.data.x is None:
        global_dataset.transform = Constant_Long(0)

    # We loop over all combinations of convergence and max_iterations
    for convergence in WL_CONVERGENCE:

        for k_wl in WL_MAX_ITERATIONS:
            seed_everything(SEED)

            # Create dataset
            dataset = Wrapper_WL_TUDataset(global_dataset, k_wl=k_wl, wl_convergence=convergence).shuffle()
            wl_conv = dataset.wl_conv

            if k_wl > 0:
                x = torch.stack([wl_conv.histogram(data.x).squeeze() for data in dataset], dim=0)
            else:
                # Check if the original dataset has node features: 1. Case Features are one-hot encoded 2. Case Features are not one-hot encoded
                if global_dataset.transform is None:
                    x = torch.stack([data.x.sum(dim=0) for data in dataset], dim=0)
                else:
                    x = torch.stack([(data.x == 0).count_nonzero(dim=0) for data in dataset], dim=0)

            y = torch.tensor([data.y for data in dataset])
            
            max_accuracy = calculate_max_accuracy(x, y).item()
            wandb.log({"max_acc": max_accuracy, 'k_wl': k_wl, 'convergence': convergence})
            print(f"Dataset: {args.dataset}, Convergence: {convergence}, k_wl: {k_wl}, Max Accuracy: {max_accuracy}")

    wandb.finish()


if __name__ == "__main__":
    main()