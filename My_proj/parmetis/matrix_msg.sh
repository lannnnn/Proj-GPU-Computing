#!/bin/bash 

#SBATCH --partition=edu-thesis
#SBATCH --tasks=1
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --account=flavio.vella.tesi
#SBATCH --time=05:00:00 

#SBATCH --output=../datasets/small/soc-Slashdot0811/matrix_info.out    
#SBATCH --error=../datasets/small/soc-Slashdot0811/matrix_info.err

./count_dimension ~/Proj-GPU-Computing/My_proj/datasets/small/soc-Slashdot0811/soc-Slashdot0811.mtx