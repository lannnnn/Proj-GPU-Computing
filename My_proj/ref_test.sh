#!/bin/bash 

#SBATCH --partition=edu5
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
    
#SBATCH --output=ref_block_test.out     
#SBATCH --error=ref_block_test.err     

module load cuda/12.1

#srun ./ref_block_cuda -f ./data/unweighted/0_mycielskian13.el -t 0.2 -b 8 -e -l
#srun ./ref_block_cuda -f ./data/unweighted/ca-HepPh_r.el -t 0.2 -b 8 -e -l
#srun ./ref_block_cuda -f ./data/unweighted/cs_department.el -t 0.2 -b 8 -e -l
#srun ./ref_block_cuda -f ./data/unweighted/ia-wikiquote-user-edits-nodup.el -t 0.2 -b 8 -e -l
#srun ./ref_block_cuda -f ./data/unweighted/seventh_graders.el -t 0.2 -b 8 -e -l
#srun ./ref_block_cuda -f ./data/unweighted/social_location.el -t 0.01 -b 8 -e 
#srun ./ref_block_cuda -f ./data/unweighted/twitter.el -t 0.2 -b 8 -e
#srun ./ref_block_cuda -f ./data/unweighted/wiki-Vote_r.el -t 0.2 -b 8 -e -l
#srun ./ref_block_cuda -f ./data/weighted/1_nemeth21.el -t 0.2 -b 8 -e -l
#srun ./ref_block_cuda -f ./data/weighted/2_nemeth22.el -t 0.2 -b 8 -e -l
#srun ./ref_block_cuda -f ./data/weighted/3_TSC.el -t 0.2 -b 8 -e -l
#srun ./ref_block_cuda -f ./data/weighted/bcsstk18_r.el -t 0.2 -b 8 -e -l

#srun ./ref_block_cuda -f ./data/unweighted/grid2.mtx -t 0.2 -b 8 -m -l
#srun ./ref_block_cuda -f ./data/weighted/494_bus.mtx -t 0.2 -b 8 -m -l
#srun ./ref_block_cuda -f ./data/weighted/1138_bus.mtx -t 0.2 -b 8 -m -l
#srun ./ref_block_cuda -f ./data/weighted/freeFlyingRobot_4.mtx -t 0.2 -b 8 -m -l
#srun ./ref_block_cuda -f ./data/weighted/freeFlyingRobot_7.mtx -t 0.2 -b 8 -m -l
#srun ./ref_block_cuda -f ./data/weighted/kron_g500-logn16.mtx -t 0.2 -b 8 -m -l
srun ./ref_block_cuda -f ./data/unweighted/cs_department.el -t 0.9 -b 64 -e