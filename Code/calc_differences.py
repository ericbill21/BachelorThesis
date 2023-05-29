import math

import torch

import wandb

SWEEP_IDS = ["zaffc3vo", "lf4lhg08", "gykfim06", "eh0fsrx1"]
PERCENTAGE_STEPS = [0.5, 0.25, 0.1, 0.05, 0.01]

api = wandb.Api()

wlnn_val_acc = []
wlnn_train_acc = []

gnn_val_acc = []
gnn_train_acc = []
for sweep_id in SWEEP_IDS:

    print(f"Fetching sweep with id {sweep_id}...\n")
    sweep = api.sweep(f"eric-bill/BachelorThesis/sweeps/id/{sweep_id}")

    run_list = list(sweep.runs)
    num_runs = len(run_list)
    iter = 0

    for run in run_list:
        print(f"Run number {iter}/{num_runs}", end="\r")
        
        history_data = run.history()

        try:
            row_index = history_data['val_acc'].argmax()
            best_val_acc = history_data['val_acc'][row_index]
            best_train_acc = history_data['train_acc'][row_index]
        except:
            print(f"\nRun {run.name} with run id {run.id} has no val_acc metric.\n")
            continue

        if run.name.startswith('1WL+NN'):
            wlnn_val_acc.append(best_val_acc)
            wlnn_train_acc.append(best_train_acc)
        elif run.name.startswith('GIN'):
            gnn_val_acc.append(best_val_acc)
            gnn_train_acc.append(best_train_acc)
        else:
            print(f"\nRun {run.name} is not a valid run name with run id {run.id}.\n")

        iter += 1


print(f"Found {len(wlnn_val_acc)} 1WL+NN models and {len(gnn_val_acc)} GIN models.")
wlnn_val_acc = torch.tensor(wlnn_val_acc)
wlnn_train_acc = torch.tensor(wlnn_train_acc)

gnn_val_acc = torch.tensor(gnn_val_acc)
gnn_train_acc = torch.tensor(gnn_train_acc)

# Sorting descending
sorting_indices = torch.argsort(wlnn_val_acc, descending=True)
wlnn_val_acc = wlnn_val_acc[sorting_indices]
wlnn_train_acc = wlnn_train_acc[sorting_indices]

sorting_indices = torch.argsort(gnn_val_acc, descending=True)
gnn_val_acc = gnn_val_acc[sorting_indices]
gnn_train_acc = gnn_train_acc[sorting_indices]

# Calculating mean difference between validation and train accuracy
data = []
for percentage in PERCENTAGE_STEPS:
    # Creating new row for table
    new_row = [f"{int(percentage * 100)}%"]
    
    # Calculating mean difference between validation and train accuracy for 1WL+NN
    best_wlnn_val_acc = wlnn_val_acc[:math.ceil(len(wlnn_val_acc) * percentage)]
    best_wlnn_train_acc = wlnn_train_acc[:math.ceil(len(wlnn_train_acc) * percentage)]

    mean_diff_wlnn = torch.mean(best_wlnn_train_acc - best_wlnn_val_acc)
    new_row.append(mean_diff_wlnn.item())

    # Calculating mean difference between validation and train accuracy for GIN
    best_gnn_val_acc = gnn_val_acc[:math.ceil(len(gnn_val_acc) * percentage)]
    best_gnn_train_acc = gnn_train_acc[:math.ceil(len(gnn_train_acc) * percentage)]

    mean_diff_gnn = torch.mean(best_gnn_train_acc - best_gnn_val_acc)
    new_row.append(mean_diff_gnn.item())

    # Calculating difference between 1WL+NN and GIN
    new_row.append(mean_diff_wlnn.item() - mean_diff_gnn.item())

    data.append(new_row)

wandb.init(project="BachelorThesis", entity="eric-bill", name="Best models difference")

columns = ["Best Models", "1WL+NN", "GIN", "Difference"]
table = wandb.Table(data=data, columns=columns)
wandb.log({"Best_Models_Difference": table})

wandb.finish()