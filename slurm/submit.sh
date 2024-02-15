# (submit.sh)
#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH -q gpu
#SBATCH -p gpu
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1   # This needs to match Trainer(devices=...)
#SBATCH --gmem=20G
#SBATCH --time=0-02:00:00

# activate conda env
# source activate $1

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo

# might need the latest CUDA
# module load NCCL/2.4.7-1-cuda.10.0

# run script from above
srun python3 __main__.py --config config/pretrain.yaml --trainer.devices 1


# 90 seconds before training ends
SBATCH --signal=SIGUSR1@90
