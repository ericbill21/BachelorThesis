import wandb
import random

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10,
    }
)

wandb.define_metric("epoch")
# set all other train/ metrics to use this step
wandb.define_metric("Loss:*", step_metric="epoch")

# simulate training
arr1 = [random.random()*2 - 1 + i for i in range(10)]
print(arr1)

arr2 = [random.random()*2 - 1 + i for i in range(10)]
print(arr2)
for i, item in enumerate(arr1):
    wandb.log({'Loss:1' : item, 'epoch': i})

for i, item in enumerate(arr2):
    wandb.log({'Loss:2' : item, 'epoch': i})

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()