#!/bin/bash 

#SBATCH --partition=edu5
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
    
#SBATCH --output=block_test.out     
#SBATCH --error=block_test.err     

module load cuda/12.1

srun ./block_cuda