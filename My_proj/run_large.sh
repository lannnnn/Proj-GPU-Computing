#!/bin/bash 

#SBATCH --partition=edu-thesis
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --account=flavio.vella.tesi
#SBATCH --time=05:00:00
    
#SBATCH --output=datasets/medium/com-Amazon/ref_1/com-Amazon_0.2.out     
#SBATCH --error=datasets/medium/com-Amazon/ref_1/com-Amazon_0.2.err     

#module load cuda/12.1

#srun ./ref_block_cuda_128 -f ./datasets/toy/wiki-Vote/wiki-Vote.mtx -l -b 64 -m -o ./datasets/toy/wiki-Vote/ref_128/unmask.reorder
#srun ./ref_block_cuda_128 -f ./datasets/small/email-Enron/email-Enron.mtx -l -b 64 -m -o ./datasets/small/email-Enron/ref_128/unmask.reorder
srun ./ref_block_cuda_1_mask -f ./datasets/medium/com-Amazon/com-Amazon.mtx -t 0.2 -b 64 -m -o ./datasets/medium/com-Amazon/ref_1/com-Amazon.reorder

#srun ./blocktest -f ./datasets/toy/wiki-Vote/wiki-Vote.mtx -l -b 64 -m -o ./datasets/toy/wiki-Vote/cpu/unmask.reorder
#srun ./blocktest -f ./datasets/small/soc-Slashdot0811/soc-Slashdot0811.mtx -l -b 64 -m -o ./datasets/small/soc-Slashdot0811/cpu/unmask.reorder
#srun ./blocktest_mask -f ./datasets/medium/web-Stanford/web-Stanford.mtx -t 0.01 -b 64 -m -o ./datasets/medium/web-Stanford/cpu/web-Stanford.reorder