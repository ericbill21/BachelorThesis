import wandb
import torch
api = wandb.Api()

run = api.run("eric-bill/BachelorThesis/9qkb8bbi")

k_fold = run.config['k-fold']
epochs = run.config['Epochs']
history = run.scan_history()

mean_train_acc = torch.tensor([[row[f'train accuracy: fold {fold+1}'] for row in history] for fold in range(k_fold)], dtype=torch.float32).mean(dim=0)
mean_train_loss = torch.tensor([[row[f'train loss: fold {fold+1}'] for row in history] for fold in range(k_fold)], dtype=torch.float32).mean(dim=0)
mean_val_acc = torch.tensor([[row[f'val accuracy: fold {fold+1}'] for row in history] for fold in range(k_fold)], dtype=torch.float32).mean(dim=0)
mean_val_loss = torch.tensor([[row[f'val loss: fold {fold+1}'] for row in history] for fold in range(k_fold)], dtype=torch.float32).mean(dim=0)

wandb.define_metric("train accuracy", summary="last", step_metric="epoch")
wandb.define_metric("val accuracy", summary="last", step_metric="epoch")
wandb.define_metric("train loss", summary="last", step_metric="epoch")
wandb.define_metric("val loss", summary="last", step_metric="epoch")

for i in range(epochs):
    wandb.log({"epoch": i+1, "train accuracy": mean_train_acc[i], "val accuracy": mean_val_acc[i], "train loss": mean_train_loss[i], "val loss": mean_val_loss[i]})

run.update()