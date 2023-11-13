#!/bin/bash 

#SBATCH --partition=edu-thesis
#SBATCH --tasks=1
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --account=flavio.vella.tesi
#SBATCH --time=05:00:00 

#SBATCH --output=../datasets/convert/small/mk12-b4/mk12-b4_0.9.out    
#SBATCH --error=../datasets/convert/small/mk12-b4/mk12-b4_0.9.err 
    
module load cuda/12.1

#gpmetis ~/Proj-GPU-Computing/My_proj/datasets/convert/toy/ca-AstroPh/ca-AstroPh.graph 13111
#gpmetis ~/Proj-GPU-Computing/My_proj/datasets/convert/toy/lp_cre_a/lp_cre_a.graph 768
#gpmetis ~/Proj-GPU-Computing/My_proj/datasets/convert/toy/inf-power/inf-power.graph 588
#gpmetis ~/Proj-GPU-Computing/My_proj/datasets/convert/toy/ca-HepTh/ca-HepTh.graph 4378
#gpmetis ~/Proj-GPU-Computing/My_proj/datasets/convert/toy/Franz4/Franz4.graph 18
#gpmetis ~/Proj-GPU-Computing/My_proj/datasets/convert/toy/p2p-Gnutella05/p2p-Gnutella05.graph 3359
#gpmetis ~/Proj-GPU-Computing/My_proj/datasets/convert/toy/wiki-Vote/wiki-Vote.graph 3709
gpmetis ~/Proj-GPU-Computing/My_proj/datasets/convert/small/mk12-b4/mk12-b4.graph 169

#nvidia-smi
