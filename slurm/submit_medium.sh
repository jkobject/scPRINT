#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH -q gpu
#SBATCH -p gpu
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:A40:1,gmem:48G
#SBATCH --time=0-12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=8G
#SBATCH --tasks-per-node=1

# conda activate scprint17
# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo
#      # This needs to match Trainer(devices=...)

#module load cuda/11.7
#module load cudnn/11.x-v8.7.0.84

#lamin load scprint

# run script from above
srun python3 scprint/__main__.py fit --trainer.logger.offline True --data.num_workers 16 --model.lr 0.002 --config config/pretrain_small.yaml 

# 90 seconds before training ends
SBATCH --signal=SIGUSR1@90
