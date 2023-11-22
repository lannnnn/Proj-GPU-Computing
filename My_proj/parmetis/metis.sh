#!/bin/bash 

#SBATCH --partition=edu-thesis
#SBATCH --tasks=1
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --account=flavio.vella.tesi
#SBATCH --time=05:00:00 

#SBATCH --output=../datasets/convert/small/loc-Brightkite/loc-Brightkite_128.out    
#SBATCH --error=../datasets/convert/small/loc-Brightkite/loc-Brightkite_128.err 
    
module load cuda/12.1

#gpmetis ~/Proj-GPU-Computing/My_proj/datasets/convert/toy/ca-AstroPh/ca-AstroPh.graph 16
#gpmetis ~/Proj-GPU-Computing/My_proj/datasets/convert/toy/lp_cre_a/lp_cre_a.graph 128
#gpmetis ~/Proj-GPU-Computing/My_proj/datasets/convert/toy/inf-power/inf-power.graph 16
#gpmetis ~/Proj-GPU-Computing/My_proj/datasets/convert/toy/ca-HepTh/ca-HepTh.graph 128
#gpmetis ~/Proj-GPU-Computing/My_proj/datasets/convert/toy/Franz4/Franz4.graph 16
#gpmetis ~/Proj-GPU-Computing/My_proj/datasets/convert/toy/p2p-Gnutella05/p2p-Gnutella05.graph 128
#gpmetis ~/Proj-GPU-Computing/My_proj/datasets/convert/toy/wiki-Vote/wiki-Vote.graph 16
gpmetis ~/Proj-GPU-Computing/My_proj/datasets/convert/small/loc-Brightkite/loc-Brightkite.graph 128
#gpmetis ~/Proj-GPU-Computing/My_proj/datasets/convert/medium/com-Amazon/com-Amazon.graph 16

#nvidia-smi
#16,  64,/ 128 
