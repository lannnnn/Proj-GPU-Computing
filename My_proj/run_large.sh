#!/bin/bash 

#SBATCH --partition=edu-thesis
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --account=flavio.vella.tesi
#SBATCH --time=05:00:00
    
#SBATCH --output=result/large/twitter_priority_16.out     
#SBATCH --error=result/large/twitter_priority_16.err     

#module load cuda/12.1

#srun ./ref_block_cuda -f ./data/large/kron_g500-logn16.mtx -t 0.2 -b 64 -m -l
#srun ./ref_block_cuda -f ./data/large/twitter.el -t 0.2 -b 64 -e -l
#srun ./ref_block_cuda -f ./data/large/t2em.mtx -t 0.2 -b 64 -m -l
#srun ./ref_block_cuda -f ./data/large/apache2.mtx -t 0.2 -b 64 -m -l
#srun ./ref_block_cuda -f ./data/large/rajat29.mtx -t 0.2 -b 64 -m -l
#srun ./ref_block_cuda -f ./data/large/delaunay_n20.mtx -t 0.6 -b 64 -m 

srun ./ref_block_cuda -f ./data/large/twitter.el -t 0.9 -b 64 -e -l

#srun ./blocktest -f ./data/large/twitter.el -t 0.9 -b 64 -e