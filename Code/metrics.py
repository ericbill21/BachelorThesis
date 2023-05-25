import warnings

import torchmetrics

import wandb


def create_metrics_dict(metric_names, num_classes, IS_CLASSIFICATION):
    metric_func = {}
    for metric_name in metric_names:
        if metric_name == "f1_score" and IS_CLASSIFICATION:
            wandb.define_metric(f"val_f1_score: fold*", step_metric="epoch")
            wandb.define_metric(f"val_f1_score", summary="max", step_metric="epoch")

            f1_score = torchmetrics.classification.F1Score(
                num_labels=num_classes,
                average="macro",
                task="multiclass" if num_classes > 2 else "binary",
            )
            metric_func["f1_score"] = f1_score

        elif metric_name == "roc_auc_score" and IS_CLASSIFICATION:
            wandb.define_metric(f"val_roc_auc_score: fold*", step_metric="epoch")
            wandb.define_metric(f"val_roc_auc_score", summary="max", step_metric="epoch")

            roc_auc_score = torchmetrics.classification.AUROC(
                num_labels=num_classes,
                average="macro",
                task="multiclass" if num_classes > 2 else "binary",
            )
            metric_func["roc_auc_score"] = roc_auc_score

        else:
            warnings.warn(
                f"Metric {metric_name} is either supported for this dataset or not yet implemented."
            )
    
    return metric_func