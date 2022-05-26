#!/bin/bash
## Add here your SBATCH config.
#SBATCH -N 8
#SBATCH --gres=gpu:8
#SBATCH -c 2
#SBATCH --ntasks-per-node 8
#SBATCH --time=72:00:00

# Collect the address of the master node.
export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

# Execute application command. Modify the command script accordingly.
srun ./ddp_run.sh
