#!/bin/bash 

#SBATCH --partition=edu-thesis
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --account=flavio.vella.tesi
#SBATCH --time=05:00:00
    
#SBATCH --output=datasets/medium/amazon0312/cpu/amazon0312_0.5.out     
#SBATCH --error=datasets/medium/amazon0312/cpu/amazon0312_0.5.err     

#module load cuda/12.1

#srun ./ref_block_cuda_128 -f ./datasets/medium/web-Stanford/web-Stanford.mtx -l -b 64 -m -o ./datasets/medium/web-Stanford/ref_128/web-Stanford_ref_128.reorder

#srun ./blocktest -f ./datasets/toy/wiki-Vote/wiki-Vote.mtx -l -b 64 -m -o ./datasets/toy/wiki-Vote/cpu/wiki-Vote.reorder
#srun ./blocktest -f ./datasets/small/soc-Slashdot0811/soc-Slashdot0811.mtx -l -b 64 -m -o ./datasets/small/soc-Slashdot0811/cpu/soc-Slashdot0811.reorder
srun ./blocktest -f ./datasets/medium/amazon0312/amazon0312.mtx -t 0.5 -b 64 -m -o ./datasets/medium/amazon0312/cpu/amazon0312.reorder