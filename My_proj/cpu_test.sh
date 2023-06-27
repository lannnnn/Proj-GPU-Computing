#!/bin/bash 

#SBATCH --partition=edu5
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
    
#SBATCH --output=cpu_test.out     
#SBATCH --error=cpu_test.err     

#module load cuda/12.1

srun ./blocktest -f ./data/unweighted/wiki-Vote_r.el -t 0.6 -b 8