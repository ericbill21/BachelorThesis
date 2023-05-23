import argparse
import time
import warnings

import numpy as np
import torch
import torch_geometric
import torchmetrics
import utils
from models import load_model
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.transforms import OneHotDegree, ToDevice
from utils import Constant_Long, WL_Transformer, Wrapper_TUDataset

import wandb

# GLOBAL VARIABLES
LOG_INTERVAL = 50
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {DEVICE}")

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch GNN')
parser.add_argument('--dataset', type=str, default='PROTEINS', help='Dataset name.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=32, help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=0.02, help='Initial learning rate.')
parser.add_argument('--k_fold', type=int, default=10, help='Number of folds for k-fold cross validation.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--k_wl', type=int, default=3, help='Number of Weisfeiler-Lehman iterations, or if -1 it runs until convergences.')
parser.add_argument('--model', type=str, default='1WL+NN:Embedding-Sum', help='Model to use. Options are "1WL+NN:Embedding-{SUM,MAX,MEAN}" or "GIN:{SUM,MAX,MEAN}".')
parser.add_argument('--wl_convergence', type=str, choices=['True','False'], help='Whether to use the convergence criterion for the Weisfeiler-Lehman algorithm.')
parser.add_argument('--tags', nargs='+', default=[], help='Tags for the run on wandb.')
parser.add_argument('--loss_func', type=str, default='CrossEntropyLoss', help='Loss function to use. Options are "L1Loss", "MSELoss", "CrossEntropyLoss", "CTCLoss", "NLLLoss", "PoissonNLLLoss", "KLDivLoss", "BCELoss", "BCEWithLogitsLoss", "MarginRankingLoss", "HingeEmbeddingLoss", "MultiLabelMarginLoss", "SmoothL1Loss", "SoftMarginLoss", "MultiLabelSoftMarginLoss", "CosineEmbeddingLoss", "MultiMarginLoss", "TripletMarginLoss" and "TripletMarginWithDistanceLoss".')
parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer to use. Options are "SGD", "Adam", "Adadelta", "Adagrad", "Adamax", "RMSprop" and "Rprop".')
parser.add_argument('--metric', nargs='+', default=[], help='Metric to use. Options are "f1_score" and "roc_auc_score".')
parser.add_argument('--transformer', type=str, default='None', help='Transformer to use. Options are "OneHotDegree", "Constant_Long" and "None".')
parser.add_argument('--transformer_args', nargs='+', default=[], help='Arguments for the transformer. For example, for the OneHotDegree transformer, the argument is the maximum degree.')
parser.add_argument('--embedding_dim', type=int, default=8, help='Dimension of the node embeddings. Embeddings are only used for the 1WL+NN models.')
parser.add_argument('--mlp_layer_size', type=int, default=64, help='Size of the initial MLP hidden layers. The last MLP layer always has the same size as the number of classes.')
parser.add_argument('--mlp_num_layers', type=int, default=2, help='Number of MLP hidden layers.')
parser.add_argument('--gnn_layers', type=int, default=5, help='Number of GNN layers.')
parser.add_argument('--activation_func', type=str, default='relu', help='Activation function to use. Options are "relu", "leaky_relu", "elu", "gelu", "tanh", "sigmoid", "softplus", "softsign", "prelu", "rrelu", "selu", "celu", "logsigmoid", "hardsigmoid", "tanhshrink", "hardtanh", "logsoftmax", "softmin", "softmax", "softshrink", "relu6", "elu6", "silu", "mish", "swish", "hardsigmoid" and "hardswish".')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate. 0.0 means no dropout.')
parser.add_argument('--mlp_norm', type=str, default='batch_norm', help='Batch normalization to use. Options are "batch_norm" and "layer_norm".')
parser.add_argument('--jk', type=str, default='cat', help='Jumping knowledge to use. Options are "cat", "max" and "lstm".')
parser.add_argument('--gnn_hidden_channels', type=int, default=16, help='Number of hidden channels in the GNN.')
args = parser.parse_args()

# Convert arguments
args.wl_convergence = args.wl_convergence == "True"

# Set seed for reproducibility
utils.seed_everything(args.seed)

IS_CLASSIFICATION = (
    False if args.dataset in ["ZINC", "ZINC_val", "ZINC_test", "ZINC_full"] else True
)

run = wandb.init(
    project="BachelorThesis",
    name=f"{args.model}: {time.strftime('%d.%m.%Y %H:%M:%S')}",
    tags=args.tags,
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
        "mlp_num_layers": args.mlp_num_layers,
        "gnn_layers": args.gnn_layers,
        "activation_func": args.activation_func,
        "dropout": args.dropout,
        "mlp_norm": args.mlp_norm,
        "jk": args.jk,
        "gnn_hidden_channels": args.gnn_hidden_channels,
    },
)

# Define metrics
wandb.define_metric("epoch")
wandb.define_metric(f"train_loss: fold*", step_metric="epoch")
wandb.define_metric(f"val_loss: fold*", step_metric="epoch")
wandb.define_metric(f"train_loss", summary="min", step_metric="epoch")
wandb.define_metric(f"val_loss", summary="min", step_metric="epoch")
wandb.define_metric(f"train_acc: fold*", step_metric="epoch")
wandb.define_metric(f"val_acc: fold*", step_metric="epoch")
wandb.define_metric(f"train_acc", summary="max", step_metric="epoch")
wandb.define_metric(f"val_acc", summary="max", step_metric="epoch")

# Prepare Pre Dataset Transformers
transformer_list = [ToDevice(DEVICE)]
if args.model.startswith("1WL+NN"):
    wl_tranformer = WL_Transformer(
        use_node_attr=True,
        max_iterations=args.k_wl,
        check_convergence=args.wl_convergence,
        device=DEVICE,
    )
    transformer_list.append(wl_tranformer)

elif args.transformer == "OneHotDegree":
    one_hot_degree_transformer = OneHotDegree(max_degree=args.tramsformer_args[0])
    transformer_list.append(one_hot_degree_transformer)

elif args.transformer == "Constant_Long":
    constant_transformer = Constant_Long(args.transformer_args[0])
    transformer_list.append(constant_transformer)

# Load Dataset from https://chrsmrrs.github.io/datasets/docs/datasets/
dataset = Wrapper_TUDataset(
    root=f"Code/datasets",
    name=f"{args.dataset}",
    use_node_attr=False,
    pre_transform=transformer_list,
    pre_shuffle=True,
)

# Load model
model = load_model(
    model_name=args.model,
    input_dim=dataset.num_node_features,
    output_dim=dataset.num_classes,
    is_classification=True,
    largest_color=dataset.max_node_feature + 1,
    embedding_dim=args.embedding_dim,
    mlp_hidden_layer_conf=[args.mlp_layer_size] * args.mlp_num_layers,
    gnn_layers = args.gnn_layers,
    activation_func = args.activation_func,
    dropout = args.dropout,
    mlp_norm = args.mlp_norm,
    jk = args.jk,
    gnn_hidden_channels = args.gnn_hidden_channels).to(DEVICE)

# Load optimizer
optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr)

# Load loss function
loss_func = getattr(torch.nn, args.loss_func)()

metric_func = {}
for metric_name in args.metric:
    if metric_name == "f1_score" and IS_CLASSIFICATION:
        wandb.define_metric(f"val_f1_score: fold*", step_metric="epoch")
        wandb.define_metric(f"val_f1_score", summary="max", step_metric="epoch")

        f1_score = torchmetrics.classification.F1Score(
            num_labels=dataset.num_classes,
            average="macro",
            task="multiclass" if dataset.num_classes > 2 else "binary",
        )
        metric_func["f1_score"] = f1_score

    elif metric_name == "roc_auc_score" and IS_CLASSIFICATION:
        wandb.define_metric(f"val_roc_auc_score: fold*", step_metric="epoch")
        wandb.define_metric(f"val_roc_auc_score", summary="max", step_metric="epoch")

        roc_auc_score = torchmetrics.classification.AUROC(
            num_labels=dataset.num_classes,
            average="macro",
            task="multiclass" if dataset.num_classes > 2 else "binary",
        )
        metric_func["roc_auc_score"] = roc_auc_score

    else:
        warnings.warn(
            f"Metric {metric_name} is either supported for this dataset or not yet implemented."
        )

# Log the model to wandb
wandb.watch(model, log="all")

# Use Stratified K-Fold cross validation if it is a classification task
cross_validation = StratifiedKFold(
    n_splits=args.k_fold, shuffle=True, random_state=args.seed
)
splitting_indices = list(
    cross_validation.split(np.zeros(dataset.len()), dataset.y.clone().detach().cpu())
)  # Ugly workaround for CUDA

# Initialize local variables for local logging
mean_train_acc = torch.zeros(args.epochs)
mean_val_acc = torch.zeros(args.epochs)
mean_train_loss = torch.zeros(args.epochs)
mean_val_loss = torch.zeros(args.epochs)

metric_logs = {}
for metric_name in metric_func.keys():
    metric_logs[metric_name] = torch.zeros(args.epochs)

# TRAINING LOOP: Loop over the args.k_fold splits
for fold, (train_ids, test_ids) in enumerate(splitting_indices):
    print(f"Cross-Validation Split {fold+1}/{args.k_fold}:")

    # Reset the model parameters
    model.reset_parameters()

    # Initialize the data loaders
    train_loader = PyGDataLoader(
        dataset[train_ids], batch_size=args.batch_size, shuffle=True
    )
    val_loader = PyGDataLoader(
        dataset[test_ids], batch_size=args.batch_size, shuffle=False
    )

    # Train the model
    for epoch in range(args.epochs):
        start = time.time()

        # Train, validate and test the model
        train_loss, train_acc = utils.train(
            model, train_loader, optimizer, loss_func, DEVICE
        )
        val_loss, val_acc, metric_results = utils.val(
            model, val_loader, loss_func, DEVICE, metric_func
        )

        # Log the results to wandb
        wandb.log(
            {
                f"val_loss: fold{fold+1}": val_loss,
                f"val_acc: fold{fold+1}": val_acc,
                f"train_loss: fold{fold+1}": train_loss,
                f"train_acc: fold{fold+1}": train_acc,
                "epoch": epoch + 1,
            }
        )

        # Log the results locally
        mean_train_acc[epoch] += train_acc
        mean_val_acc[epoch] += val_acc
        mean_train_loss[epoch] += train_loss
        mean_val_loss[epoch] += val_loss

        for metric_name, result in metric_results.items():
            wandb.log({f"val_{metric_name}: fold{fold+1}": result, "epoch": epoch + 1})
            metric_logs[metric_name][epoch] += result

        # Print current status
        if (epoch + 1) % LOG_INTERVAL == 0:
            print(
                f"\tEpoch: {epoch+1},\t Train Loss: {round(train_loss, 5)},"
                f"\t Train Acc: {round(train_acc, 1)}%,\t Val Loss: {round(val_loss, 5)},"
                f"\t Val Acc: {round(val_acc, 1)}%"
            )

# Averaging the local logging variables
mean_train_acc /= args.k_fold
mean_val_acc /= args.k_fold
mean_train_loss /= args.k_fold
mean_val_loss /= args.k_fold

for metric_name in metric_logs.keys():
    metric_logs[metric_name] /= args.k_fold

# Log the results to wandb
for epoch in range(args.epochs):
    wandb.log(
        {
            f"val_loss": mean_val_loss[epoch],
            f"val_acc": mean_val_acc[epoch],
            f"train_loss": mean_train_loss[epoch],
            f"train_acc": mean_train_acc[epoch],
            "epoch": epoch + 1,
        }
    )

    for metric_name, metric_res in metric_logs.items():
        wandb.log({f"val_{metric_name}": metric_res[epoch], "epoch": epoch + 1})

wandb.finish()
