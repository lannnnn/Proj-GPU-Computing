#!/bin/bash 

#SBATCH --partition=edu-thesis
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --account=flavio.vella.tesi
#SBATCH --time=05:00:00
    
#SBATCH --output=datasets/medium/Chevron3/cpu/unmask.out     
#SBATCH --error=datasets/medium/Chevron3/cpu/unmask.err     

#module load cuda/12.1

#srun ./ref_block_cuda_128 -f ./datasets/medium/Chevron3/Chevron3.mtx -l -b 64 -m -o ./datasets/medium/Chevron3/ref_128/Chevron3_ref_128.reorder

#srun ./blocktest -f ./datasets/toy/GT01R/GT01R.mtx -l -b 64 -m -o ./datasets/toy/GT01R/cpu/GT01R.reorder
#srun ./blocktest -f ./datasets/small/RFdevice/RFdevice.mtx -l -b 64 -m -o ./datasets/small/RFdevice/cpu/RFdevice.reorder
srun ./blocktest -f ./datasets/medium/Chevron3/Chevron3.mtx -t 0.5 -b 64 -m -o ./datasets/medium/Chevron3/cpu/unmask.reorder