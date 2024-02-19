#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH -q gpu
#SBATCH -p gpu
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:A100:2,gmem:30G
#SBATCH --time=0-02:00:00
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --tasks-per-node=2

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
srun python3 scprint/__main__.py fit --config config/pretrain.yaml --trainer.devices 2 --trainer.logger.offline True --data.num_workers 8 --trainer.strategy ddp_find_unused_parameters_true

# 90 seconds before training ends
SBATCH --signal=SIGUSR1@90
