#!/bin/bash 

#SBATCH --partition=edu-thesis
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --account=flavio.vella.tesi
#SBATCH --time=05:00:00
    
#SBATCH --output=datasets/toy/inf-power/cpu/inf-power_cpu.out     
#SBATCH --error=datasets/toy/inf-power/cpu/inf-power_cpu.err   

#module load cuda/12.1

#srun ./ref_block_cuda_128 -f ./datasets/small/email-Enron/email-Enron.mtx -l -b 64 -m -o ./datasets/small/email-Enron/ref_128/email-Enron_ref_128.reorder

srun ./blocktest -f ./datasets/toy/inf-power/inf-power.mtx -l -b 64 -m -o ./datasets/toy/inf-power/cpu/inf-power_cpu.reorder