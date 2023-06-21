#!/usr/bin/zsh

### Job name
#SBATCH --job-name=Erics_1WLNN

### Output path for stdout and stderr
### %J is the job ID, %I is the array ID
#SBATCH --output=output_%J.txt

### Ask for 10 GB memory
#SBATCH --mem-per-cpu=32G   #M is the default and can therefore be omitted, but could also be K(ilo)|G(iga)|T(era)

### Request 4 CPU core
#SBATCH --cpus-per-task=2

### Request the time you need for execution. The full format is D-HH:MM:SS
### You must at least specify minutes OR days and hours and may add or
### leave out any other parameters
#SBATCH --time=1200

### Name the job
#SBATCH --job-name=SERIAL_JOB

### Request a host with a Volta GPU
### If you need two GPUs, change the number accordingly
#SBATCH --gres=gpu:volta:1

### if needed: switch to your working directory (where you saved your program)
#cd /home/cc147794/BachelorThesis/

### Load modules
module load cuDNN/8.6.0.163-CUDA-11.8.0

### Activate python environment
# Insert this AFTER the #SLURM argument section of your job script
export CONDA_ROOT=$HOME/miniconda3
. $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
conda activate base

### Begin of executable commands
python Code/main.py --batch_size=32 --dataset=ENZYMES --encoding_kwargs="{'embedding_dim': 128}" --k_fold=10 --k_wl=1 --lr=0.004805446649227572 --max_epochs=200 --mlp_kwargs="{'act': 'relu', 'dropout': 0.11295853328017604, 'norm': 'batch_norm', 'num_layers': 4}" --model=1WL+NN:Embedding-Max --num_repition=5 --seed=42 --tags=replicate_run2 --wl_convergence=False
python Code/main.py --batch_size=32 --dataset=ENZYMES --encoding_kwargs="{'embedding_dim': 64}" --k_fold=10 --k_wl=1 --lr=0.006829110185485606 --max_epochs=200 --mlp_kwargs="{'act': 'relu', 'dropout': 0.05201037869129854, 'norm': 'batch_norm', 'num_layers': 4}" --model=1WL+NN:Embedding-Mean --num_repition=5 --seed=42 --tags=replicate_run2 --wl_convergence=False
python Code/main.py --batch_size=32 --dataset=ENZYMES --encoding_kwargs="{'embedding_dim': 128}" --k_fold=10 --k_wl=1 --lr=0.003733873412428764 --max_epochs=200 --mlp_kwargs="{'act': 'relu', 'dropout': 0.0980200557864032, 'norm': 'batch_norm', 'num_layers': 3}" --model=1WL+NN:Embedding-Sum --num_repition=5 --seed=42 --tags=replicate_run2 --wl_convergence=False
python Code/main.py --batch_size=32 --dataset=ENZYMES --gnn_kwargs="{'act': 'relu', 'act_first': False, 'dropout': 0.015586012941200256, 'hidden_channels': 64, 'jk': 'cat', 'norm': None, 'num_layers': 5}" --k_fold=10 --k_wl=1 --lr=0.0059926905368882285 --max_epochs=200 --mlp_kwargs="{'act': 'relu', 'dropout': 0.08701262870625089, 'norm': 'batch_norm', 'num_layers': 2}" --model=GCN:Max --num_repition=5 --seed=42 --tags=replicate_run2 --wl_convergence=False
python Code/main.py --batch_size=32 --dataset=ENZYMES --gnn_kwargs="{'act': 'relu', 'act_first': False, 'dropout': 0.0248244912027094, 'hidden_channels': 64, 'jk': 'cat', 'norm': None, 'num_layers': 5}" --k_fold=10 --k_wl=1 --lr=0.0025150587247895853 --max_epochs=200 --mlp_kwargs="{'act': 'relu', 'dropout': 0.004819588508462003, 'norm': 'batch_norm', 'num_layers': 4}" --model=GIN:Mean --num_repition=5 --seed=42 --tags=replicate_run2 --wl_convergence=False
python Code/main.py --gnn_kwargs="{'act': 'relu', 'act_first': False, 'dropout': 0.12270891333491817, 'hidden_channels': 64, 'jk': 'cat', 'norm': None, 'num_layers': 5}" --k_fold=10 --k_wl=1 --lr=0.002699855684273748 --max_epochs=200 --mlp_kwargs="{'act': 'relu', 'dropout': 0.06069166552982977, 'norm': 'batch_norm', 'num_layers': 3}" --model=GAT:Sum --num_repition=5 --seed=42 --tags=replicate_run2 --wl_convergence=False
python Code/main.py --batch_size=32 --dataset=IMDB-BINARY --encoding_kwargs="{'embedding_dim': 16}" --k_fold=10 --k_wl=1 --lr=0.043205812694749594 --max_epochs=200 --mlp_kwargs="{'act': 'relu', 'dropout': 0.12818714711632634, 'norm': 'batch_norm', 'num_layers': 2}" --model=1WL+NN:Embedding-Max --num_repition=5 --seed=42 --tags=replicate_run2 --wl_convergence=False
python Code/main.py --batch_size=32 --dataset=IMDB-BINARY --encoding_kwargs="{'embedding_dim': 16}" --k_fold=10 --k_wl=1 --lr=0.011039681685077896 --max_epochs=200 --mlp_kwargs="{'act': 'relu', 'dropout': 0.19977822025728648, 'norm': 'batch_norm', 'num_layers': 3}" --model=1WL+NN:Embedding-Mean --num_repition=5 --seed=42 --tags=replicate_run2 --wl_convergence=False
python Code/main.py --batch_size=32 --dataset=IMDB-BINARY --encoding_kwargs="{'embedding_dim': 16}" --k_fold=10 --k_wl=1 --lr=0.036295544195206385 --max_epochs=200 --mlp_kwargs="{'act': 'relu', 'dropout': 0.13678147258933218, 'norm': 'batch_norm', 'num_layers': 2}" --model=1WL+NN:Embedding-Sum --num_repition=5 --seed=42 --tags=replicate_run2 --wl_convergence=False
python Code/main.py --batch_size=32 --dataset=IMDB-BINARY --gnn_kwargs="{'act': 'relu', 'act_first': False, 'dropout': 0.14854921150330938, 'hidden_channels': 128, 'jk': 'cat', 'norm': None, 'num_layers': 5}" --k_fold=10 --lr=0.003591831609637687 --max_epochs=200 --mlp_kwargs="{'act': 'relu', 'dropout': 0.03648628314509648, 'norm': 'batch_norm', 'num_layers': 2}" --model=GCN:Max --num_repition=5 --seed=42 --tags=replicate_run2 --wl_convergence=False
python Code/main.py --batch_size=32 --dataset=IMDB-BINARY --gnn_kwargs="{'act': 'relu', 'act_first': False, 'dropout': 0.16401526433710445, 'hidden_channels': 64, 'jk': 'cat', 'norm': None, 'num_layers': 5}" --k_fold=10 --lr=0.0007139277139039066 --max_epochs=200 --mlp_kwargs="{'act': 'relu', 'dropout': 0.14444847700248636, 'norm': 'batch_norm', 'num_layers': 5}" --model=GCN:Mean --num_repition=5 --seed=42 --tags=replicate_run2 --wl_convergence=False
python Code/main.py --batch_size=32 --dataset=IMDB-BINARY --gnn_kwargs="{'act': 'relu', 'act_first': False, 'dropout': 0.19524136110875956, 'hidden_channels': 32, 'jk': 'cat', 'norm': None, 'num_layers': 5}" --k_fold=10 --lr=0.006965629733221943 --max_epochs=200 --mlp_kwargs="{'act': 'relu', 'dropout': 0.035344157120795465, 'norm': 'batch_norm', 'num_layers': 2}" --model=GCN:Sum --num_repition=5 --seed=42 --tags=replicate_run2 --wl_convergence=False
python Code/main.py --batch_size=33 --dataset=NCI1 --encoding_kwargs="{'embedding_dim': 32}" --k_fold=10 --k_wl=3 --lr=0.026651971176285157 --max_epochs=200 --mlp_kwargs="{'act': 'relu', 'dropout': 0.19352734850793815, 'norm': 'batch_norm', 'num_layers': 2}" --model=1WL+NN:Embedding-Max --num_repition=5 --seed=42 --tags=replicate_run2 --wl_convergence=False
python Code/main.py --batch_size=33 --dataset=NCI1 --encoding_kwargs="{'embedding_dim': 64}" --k_fold=10 --k_wl=3 --lr=0.024871674248798253 --max_epochs=200 --mlp_kwargs="{'act': 'relu', 'dropout': 0.038663816953314514, 'norm': 'batch_norm', 'num_layers': 2}" --model=1WL+NN:Embedding-Mean --num_repition=5 --seed=42 --tags=replicate_run2 --wl_convergence=False
python Code/main.py --batch_size=33 --dataset=NCI1 --encoding_kwargs="{'embedding_dim': 32}" --k_fold=10 --k_wl=3 --lr=0.01329617178644451 --max_epochs=200 --mlp_kwargs="{'act': 'relu', 'dropout': 0.12116916436590013, 'norm': 'batch_norm', 'num_layers': 2}" --model=1WL+NN:Embedding-Sum --num_repition=5 --seed=42 --tags=replicate_run2 --wl_convergence=False
python Code/main.py --batch_size=129 --dataset=NCI1 --gnn_kwargs="{'act': 'relu', 'act_first': False, 'dropout': 0.0, 'hidden_channels': 128, 'jk': 'cat', 'norm': None, 'num_layers': 5}" --k_fold=10 --k_wl=1 --lr=0.0002710194167602541 --max_epochs=200 --mlp_kwargs="{'act': 'relu', 'dropout': 0.024331698859625252, 'norm': 'batch_norm', 'num_layers': 4}" --model=GIN:Max --num_repition=5 --seed=42 --tags=replicate_run2 --wl_convergence=False
python Code/main.py --batch_size=129 --dataset=NCI1 --gnn_kwargs="{'act': 'relu', 'act_first': False, 'dropout': 4.628116958088624e-05, 'hidden_channels': 64, 'jk': 'cat', 'norm': None, 'num_layers': 5}" --k_fold=10 --lr=0.0035933769974281333 --max_epochs=200 --mlp_kwargs="{'act': 'relu', 'dropout': 0.06353618243833976, 'norm': 'batch_norm', 'num_layers': 4}" --model=GIN:Mean --num_repition=5 --seed=42 --tags=replicate_run2 --wl_convergence=False
python Code/main.py --batch_size=33 --dataset=NCI1 --gnn_kwargs="{'act': 'relu', 'act_first': False, 'dropout': 0.1995655254085076, 'hidden_channels': 16, 'jk': 'cat', 'norm': None, 'num_layers': 5}" --k_fold=10 --lr=0.01002693091446477 --max_epochs=200 --mlp_kwargs="{'act': 'relu', 'dropout': 0.18837594518660128, 'norm': 'batch_norm', 'num_layers': 2}" --model=GIN:Sum --num_repition=5 --seed=42 --tags=replicate_run2 --wl_convergence=False
python Code/main.py --batch_size=32 --dataset=PROTEINS --encoding_kwargs="{'embedding_dim': 64}" --k_fold=10 --k_wl=1 --lr=0.02390634549597559 --max_epochs=200 --mlp_kwargs="{'act': 'relu', 'dropout': 0.04824676455149229, 'norm': 'batch_norm', 'num_layers': 3}" --model=1WL+NN:Embedding-Max --num_repition=5 --tags=replicate_run2 --wl_convergence=False
python Code/main.py --batch_size=32 --dataset=PROTEINS --encoding_kwargs="{'embedding_dim': 128}" --k_fold=10 --k_wl=1 --lr=0.0007727150111472792 --max_epochs=200 --mlp_kwargs="{'act': 'relu', 'dropout': 0.04638997052917762, 'norm': 'batch_norm', 'num_layers': 3}" --model=1WL+NN:Embedding-Mean --num_repition=5 --tags=replicate_run2 --wl_convergence=False
python Code/main.py --batch_size=32 --dataset=PROTEINS --encoding_kwargs="{'embedding_dim': 16}" --k_fold=10 --k_wl=1 --lr=0.05977750052033193 --max_epochs=200 --mlp_kwargs="{'act': 'relu', 'dropout': 0.08517859950618088, 'norm': 'batch_norm', 'num_layers': 2}" --model=1WL+NN:Embedding-Sum --num_repition=5 --tags=replicate_run2 --wl_convergence=False
python Code/main.py --batch_size=32 --dataset=PROTEINS --gnn_kwargs="{'act': 'relu', 'act_first': False, 'dropout': 0.023169054612438345, 'hidden_channels': 128, 'jk': 'cat', 'norm': None, 'num_layers': 5}" --k_fold=10 --k_wl=1 --lr=0.0002710194167602541 --max_epochs=200 --mlp_kwargs="{'act': 'relu', 'dropout': 0.024331698859625252, 'norm': 'batch_norm', 'num_layers': 4}" --model=GIN:Max --num_repition=5 --seed=42 --tags=replicate_run2 --wl_convergence=False
python Code/main.py --batch_size=32 --dataset=PROTEINS --gnn_kwargs="{'act': 'relu', 'act_first': False, 'dropout': 0.16510821539249532, 'hidden_channels': 32, 'jk': 'cat', 'norm': None, 'num_layers': 5}" --k_fold=10 --k_wl=1 --lr=0.0006646494970605938 --max_epochs=200 --mlp_kwargs="{'act': 'relu', 'dropout': 0.10894450662250356, 'norm': 'batch_norm', 'num_layers': 4}" --model=GIN:Mean --num_repition=5 --seed=42 --tags=replicate_run2 --wl_convergence=False
python Code/main.py --batch_size=32 --dataset=PROTEINS --gnn_kwargs="{'act': 'relu', 'act_first': False, 'dropout': 0.08541590257837439, 'hidden_channels': 16, 'jk': 'cat', 'norm': None, 'num_layers': 5}" --k_fold=10 --k_wl=1 --lr=0.001111690949810922 --max_epochs=200 --mlp_kwargs="{'act': 'relu', 'dropout': 0.1587282961608454, 'norm': 'batch_norm', 'num_layers': 5}" --model=GCN:Sum --num_repition=5 --seed=42 --tags=replicate_run2 --wl_convergence=False