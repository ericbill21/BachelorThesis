#!/usr/bin/zsh

### Job name
#SBATCH --job-name=Erics_1WLNN

### Output path for stdout and stderr
### %J is the job ID, %I is the array ID
#SBATCH --output=output_%J.txt

### Ask for 10 GB memory
#SBATCH --mem-per-cpu=10240M   #M is the default and can therefore be omitted, but could also be K(ilo)|G(iga)|T(era)

### Request the time you need for execution. The full format is D-HH:MM:SS
### You must at least specify minutes OR days and hours and may add or
### leave out any other parameters
#SBATCH --time=480

### Name the job
#SBATCH --job-name=SERIAL_JOB

### Request a host with a Volta GPU
### If you need two GPUs, change the number accordingly
#SBATCH --gres=gpu:volta:1

### if needed: switch to your working directory (where you saved your program)
#cd $HOME/subfolder/

### Load modules
module load cuDNN/8.6.0.163-CUDA-11.8.0

### Begin of executable commands
python Code/main.py --epochs 250 --k_fold 5 --model 1WL+NN:Embedding-Sum --k_wl 3 --dataset ZINC_full --no-wl_convergence --batch_size 128