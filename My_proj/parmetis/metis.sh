#!/bin/bash 

#SBATCH --partition=edu-thesis
#SBATCH --tasks=1
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --account=flavio.vella.tesi
#SBATCH --time=05:00:00 

#SBATCH --output=../datasets/convert/toy/ca-HepTh/ca-HepTh_0.01.out    
#SBATCH --error=../datasets/convert/toy/ca-HepTh/ca-HepTh_0.01.err 
    
module load cuda/12.1

#gpmetis ~/Proj-GPU-Computing/My_proj/datasets/convert/toy/ca-AstroPh/ca-AstroPh.graph 3569
#gpmetis ~/Proj-GPU-Computing/My_proj/datasets/convert/toy/lp_cre_a/lp_cre_a.graph 3429
#gpmetis ~/Proj-GPU-Computing/My_proj/datasets/convert/toy/inf-power/inf-power.graph 3086
gpmetis ~/Proj-GPU-Computing/My_proj/datasets/convert/toy/ca-HepTh/ca-HepTh.graph 6546

#nvidia-smi
