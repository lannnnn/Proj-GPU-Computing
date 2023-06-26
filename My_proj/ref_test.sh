#!/bin/bash 

#SBATCH --partition=edu5
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
    
#SBATCH --output=ref_block_test.out     
#SBATCH --error=ref_block_test.err     

module load cuda/12.1

export CUDA_LAUNCH_BLOCKING=1

srun ./ref_block_cuda -f ./data/unweighted/wiki-Vote_r.el -t 0.8 -b 16