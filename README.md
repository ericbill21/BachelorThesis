# Bachelor's Thesis
Welcome to the GitHub repository for my bachelor's thesis titled *A Theoretical and Empirical Investigation into the Equivalence of Graph Neural Networks and the Weisfeiler-Leman Algorithm*.

Here, you will find all the code used to conduct experiments, as well as the LaTeX code for creating the written version of the thesis. Please note that we employed Weights&Biases to record our experimental results. To access these results, kindly refer to [wandb.ai/eric-bill/BachelorThesisExperiments](https://wandb.ai/eric-bill/BachelorThesisExperiments).

## Experimental Code
All the code used for the experiments can be found in the `Code` folder. Firstly, we will list the requirements needed to execute the datasets, and then we will discuss how to test the classification and regression datasets separately.

### Requirements
- `Python 3.10`
- `numpy`
- `pandas`
- `scipy`
- `sklearn`
- `torch 1.13.x`
- `torch-geometric 2.3.x`
- `wandb`

### Classification Datasets
If you want to try out a 1-WL+NN or GNN setup with any classification dataset from the TUDataset library, simplye/main.py` and provide the following arguments: run `python Code/main.py` and provide the following arguments:
- `--dataset` Name of the dataset to be tested
- `--max_epochs` Maximum number of epochs a model should be trained for
- `--batch_size` Number of samples per batch
- `--lr` Initial learning rate
- `--k_fold` Number of folds for k-fold cross-validation
- `--seed` Random seed for initializing all random samplers used
- `--k_wl` Number of Weisfeiler-Lehman iterations, or if -1 it runs until convergences
- `--model` Model to use. Options are "1WL+NN:Embedding-{Sum,Max,Mean}", "1WL+NN:{Sum,Max,Mean}", "1WL+NN:{GAT,GCN,GIN}:{Sum,Max,Mean}" or "{GAT,GCN,GIN}:{Sum,Max,Mean}"
- `--wl_convergence` {True,False} Whether to use the convergence criterion for the Weisfeiler-Lehman algorithm
- `--tags` Tags that are to be added to the recording of the run on wandb.ai
- `--num_repition` Number of repitions
- `--transformer_kwargs` Arguments for the transformer. For example, for the OneHotDegree transformer, the argument is the maximum degree.
- `--encoding_kwargs` Arguments for the encoding function. For example, for Embedding, the argument is the embedding dimension with the key "embedding_dim"
- `--mlp_kwargs` Arguments for the MLP. For example, for the MLP, the argument is the number of hidden layers with the key "num_layers"
- `--gnn_kwargs` Arguments for the GNN. For example, for GIN, the argument is the number of MLP layers with the key "num_layers"
- `--use_one_hot` {True,False} Whether to use one-hot encoding for the node features. Only for 1-WL+NN:GNN models.

### Regression Datasets
To ensure consistency in the regression datasets, we created separate Python scripts for each fixed split. To test any of these datasets, follow these steps.
1. `cd Code/`
2. `python filename` where `filename` is substituted by the following:
- `gnn_alchemy_{10K, full}` To test GNN configurations on ALCHEMY
- `gnn_zinc_{10K, full}` To test GNN configurations on ZINC
- `main_alchemy_{10K, full}` To test 1-WL+NN configurations on ALCHEMY
- `main_zinc_{10K, full}` To test 1-WL+NN configurations on ZINC

# LaTeX code
You can locate all of our LaTeX code in the `LaTeX` folder. To make it easier to write the thesis, we have separated subparts of it into individual `.tex` files. The file that brings all of these together is called `main.tex`. To compile the thesis, you only need to compile this file. 
