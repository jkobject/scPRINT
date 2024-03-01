#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH -q gpu
#SBATCH -p gpu
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:A40:4,gmem:48G
#SBATCH --time=0-02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=32G
#SBATCH --tasks-per-node=4

conda activate scprint17
# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo
#      # This needs to match Trainer(devices=...)

module load cuda/11.7
module load cudnn/11.x-v8.7.0.84

lamin load scprint

# run script from above
srun python3 scprint/__main__.py fit --config config/pretrain.yaml --trainer.logger.offline True --data.num_workers 8 --trainer.strategy ddp_find_unused_parameters_true --trainer.lr 0.002 

# 90 seconds before training ends
SBATCH --signal=SIGUSR1@90
