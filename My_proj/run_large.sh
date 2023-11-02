#!/bin/bash 

#SBATCH --partition=edu-thesis
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --account=flavio.vella.tesi
#SBATCH --time=05:00:00
    
#SBATCH --output=datasets/small/email-Enron/ref_128/email-Enron_ref_128_tau_0.7.out     
#SBATCH --error=datasets/small/email-Enron/ref_128/email-Enron_ref_128_tau_0.7.err   

#module load cuda/12.1

srun ./ref_block_cuda_128 -f ./datasets/small/email-Enron/email-Enron.mtx -t 0.7 -b 64 -m -o ./datasets/small/email-Enron/ref_128/email-Enron_ref_128_tau_0.7.reorder

#srun ./blocktest_pri -f ./datasets/toy/ca-AstroPh/ca-AstroPh.mtx -t 0.9 -b 64 -m -o ./datasets/toy/ca-AstroPh/cpu_pri/ca-AstroPh_cpu_tau_0.9.reorder