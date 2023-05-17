import time
import utils
import argparse

import wandb

import torch
import torch_geometric
from sklearn.metrics import f1_score, roc_auc_score
import torch.nn
from utils import Constant_Long, WL_Transformer
from torch_geometric.transforms import OneHotDegree, ToDevice, Compose
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np

from utils import Wrapper_TUDataset
from models import load_model

# GLOBAL VARIABLES
LOG_INTERVAL = 50
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {DEVICE}")

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch GNN')
parser.add_argument('--dataset', type=str, default='PROTEINS', help='Dataset name')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=32, help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=0.02, help='Initial learning rate.')
parser.add_argument('--k_fold', type=int, default=10, help='Number of folds for k-fold cross validation.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--k_wl', type=int, default=3, help='Number of Weisfeiler-Lehman iterations, or if -1 it runs until convergences.')
parser.add_argument('--model', type=str, default='1WL+NN:Embedding-Sum', help='Model to use.')
parser.add_argument('--wl_convergence', type=str, choices=['True','False'], help='Whether to use the convergence criterion for the Weisfeiler-Lehman algorithm.')
parser.add_argument('--tags', nargs='+', default=[])
parser.add_argument('--loss_func', type=str, default='CrossEntropyLoss', help='Loss function to use.')
parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer to use.')
parser.add_argument('--metric', nargs='+', default=[], help='Metric to use.')
parser.add_argument('--transformer', type=str, default='None', help='Transformer to use.')
parser.add_argument('--transformer_args', nargs='+', default=[], help='Arguments for the transformer.')
parser.add_argument('--embedding_dim', type=int, default=8, help='Dimension of the node embeddings.')
parser.add_argument('--mlp_layer_size', type=int, default=64, help='Size of the initial MLP hidden layers.')
parser.add_argument('--mlp_num_layers', type=int, default=2, help='Number of MLP hidden layers.')
args = parser.parse_args()

# Convert arguments
args.wl_convergence = True if args.wl_convergence == 'True' else False

# Set seed for reproducibility
utils.seed_everything(args.seed)

IS_CLASSIFICATION = False if args.dataset in ["ZINC", "ZINC_val", "ZINC_test", "ZINC_full"] else True

run = wandb.init(
    project="BachelorThesis",
    name=f"{args.model}: {time.strftime('%d.%m.%Y %H:%M:%S')}",
    tags = args.tags,
    config={
    "Epochs": args.epochs,
    "Batch size": args.batch_size,
    "Device": DEVICE,
    "k-fold": args.k_fold,
    "Dataset": args.dataset,
    "learning_rate": args.lr,
    "seed": args.seed,
    "k_wl": args.k_wl,
    "model": args.model,
    "wl_convergence": args.wl_convergence,
    "loss_func": args.loss_func,
    "optimizer": args.optimizer,
    "metric": args.metric,
    "transformer": args.transformer,
    "transformer_args": args.transformer_args,
    "embedding_dim": args.embedding_dim,
    "mlp_layer_size": args.mlp_layer_size,
    "mlp_num_layers": args.mlp_num_layers
    })

# Define metrics
wandb.define_metric("epoch")
wandb.define_metric(f"train_loss: fold*", step_metric="epoch")
wandb.define_metric(f"val_loss: fold*", step_metric="epoch")
wandb.define_metric(f"train_loss", summary="max", step_metric="epoch")
wandb.define_metric(f"val_loss", summary="max", step_metric="epoch")
wandb.define_metric(f"train_acc: fold*", step_metric="epoch")
wandb.define_metric(f"val_acc: fold*", step_metric="epoch")
wandb.define_metric(f"train_acc", summary="max", step_metric="epoch")
wandb.define_metric(f"val_acc", summary="max", step_metric="epoch")

for metric in args.metric:
    wandb.define_metric(f"val_{metric}: fold*", step_metric="epoch")
    wandb.define_metric(f"val_{metric}", summary="max", step_metric="epoch")

metric_func = []
for metric_name in args.metric:
    if metric_name == "f1_score":
        metric_func.append(lambda y_true, y_pred: f1_score(y_true, y_pred, average="micro"))
    elif metric_name == "roc_auc_score": #TODO: case when not all classes are present in the dataset
        metric_func.append(lambda y_true, y_pred: roc_auc_score(y_true, y_pred, average="micro"))
    else:
        raise NotImplementedError(f"Metric {metric_name} is not implemented.")

# Prepare Pre Dataset Transformers
transformer = [ToDevice(DEVICE)]
if args.model.startswith("1WL+NN"):
    transformer.append(WL_Transformer(use_node_attr=True, max_iterations=args.k_wl, check_convergence=args.wl_convergence))
elif args.transformer == "OneHotDegree":
    transformer.append(OneHotDegree(max_degree=args.tramsformer_args[0]))
elif args.transformer == "Constant_Long":
    transformer.append(Constant_Long(args.transformer_args[0]))

# Load Dataset from https://chrsmrrs.github.io/datasets/docs/datasets/
dataset = Wrapper_TUDataset(root=f'Code/datasets', name=f'{args.dataset}', use_node_attr=True,
                    pre_transform=Compose(transformer), pre_shuffle=True)

# Load model
model = load_model(model_name = args.model,
                    output_dim = dataset.num_classes,
                    is_classification = True,
                    device = DEVICE,
                    largest_color = transformer[-1].get_largest_color(),
                    embedding_dim = args.embedding_dim,
                    mlp_hidden_layer_conf=[args.mlp_layer_size]*args.mlp_num_layers)

# Load optimizer
optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr)

# Load loss function
loss_func = getattr(torch.nn, args.loss_func)()

# Log the model to wandb
wandb.watch(model, log="all")

# Use Stratified K-Fold cross validation if it is a classification task
cross_validation = StratifiedKFold(n_splits=args.k_fold, shuffle=True, random_state=args.seed)
splitting_indices = list(cross_validation.split(np.zeros(dataset.len()), dataset.y.clone().detach().cpu())) #Ugly workaround for CUDA

# Initialize local variables for local logging
mean_train_acc = torch.zeros(args.epochs)
mean_val_acc = torch.zeros(args.epochs)
mean_train_loss = torch.zeros(args.epochs)
mean_val_loss = torch.zeros(args.epochs)

metric_logs = []
for metric in args.metric:
    metric_logs.append(torch.zeros(args.epochs))

# TRAINING LOOP: Loop over the args.k_fold splits
for fold, (train_ids, test_ids) in enumerate(splitting_indices):
    print(f'Cross-Validation Split {fold+1}/{args.k_fold}:')

    # Reset the model parameters
    model.reset_parameters()

    # Initialize the data loaders
    train_loader = PyGDataLoader(dataset[train_ids], batch_size=args.batch_size, shuffle=True)
    val_loader = PyGDataLoader(dataset[test_ids], batch_size=args.batch_size, shuffle=False)

    # Train the model
    for epoch in range(args.epochs):
        start = time.time()

        # Train, validate and test the model
        train_loss, train_acc = utils.train(model, train_loader, optimizer, loss_func, DEVICE)
        val_loss, val_acc, metric_res = utils.val(model, val_loader, loss_func, DEVICE, metric_func)

        # Log the results to wandb
        wandb.log({f"val_loss: fold{fold+1}": val_loss,
                    f"val_acc: fold{fold+1}": val_acc,
                    f"train_loss: fold{fold+1}": train_loss,
                    f"train_acc: fold{fold+1}": train_acc,
                    "epoch": epoch+1})
        
        for metric_name, res in zip(args.metric, metric_res):
            wandb.log({f"val_{metric_name}: fold{fold+1}": res, "epoch": epoch+1})

        # Log the results locally
        mean_train_acc[epoch] += train_acc
        mean_val_acc[epoch] += val_acc
        mean_train_loss[epoch] += train_loss
        mean_val_loss[epoch] += val_loss

        for i, res in enumerate(metric_res):
            metric_logs[i][epoch] += res

        # Print current status
        if (epoch + 1) % LOG_INTERVAL == 0:
            print(f'\tEpoch: {epoch+1},\t Train Loss: {round(train_loss, 5)},' \
                    f'\t Train Acc: {round(train_acc, 1)}%,\t Val Loss: {round(val_loss, 5)},' \
                    f'\t Val Acc: {round(val_acc, 1)}%')

    
# Averaging the local logging variables
mean_train_acc /= args.k_fold
mean_val_acc /= args.k_fold
mean_train_loss /= args.k_fold
mean_val_loss /= args.k_fold

for i in range(len(metric_logs)):
    metric_logs[i] /= args.k_fold

for epoch in range(args.epochs):
    wandb.log({f"val_loss": mean_val_loss[epoch],
                f"val_acc": mean_val_acc[epoch],
                f"train_loss": mean_train_loss[epoch],
                f"train_acc": mean_train_acc[epoch],
                "epoch": epoch+1})
    
    for i, metric_res in enumerate(metric_logs):
        wandb.log({f"val_{args.metric[i]}": metric_res[epoch], "epoch": epoch+1})

wandb.finish()
