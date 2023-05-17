#!/usr/bin/env python

import wandb
import os
import multiprocessing
import collections
import random


Worker = collections.namedtuple("Worker", ("queue", "process"))
WorkerInitData = collections.namedtuple(
    "WorkerInitData", ("num", "sweep_id", "sweep_run_name", "config")
)
WorkerDoneData = collections.namedtuple("WorkerDoneData", ("val_accuracy"))



def main():
    main_run = wandb.init(project="test")
    group_name = "ttttt"

    metric = []
    for i in range(3):
        child_name = f"{main_run.name}-{i}"

        child_run = wandb.init(
            name=child_name,
            group=main_run.id,
            job_type=main_run.name,
            config=main_run.config,
        )
        val_acc = random.random()
        metric.append(val_acc)
        child_run.log({"val_accuracy": val_acc})
        child_run.finish()
    
    main_run.log({"val_accuracy": sum(metric) / len(metric)})

    main_run.finish()


if __name__ == "__main__":
    main()