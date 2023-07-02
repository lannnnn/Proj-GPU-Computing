#!/bin/bash 

#SBATCH --partition=edu5
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
    
#SBATCH --output=ref_block_test.out     
#SBATCH --error=ref_block_test.err     

module load cuda/12.1

# export CUDA_LAUNCH_BLOCKING=1

# srun ./ref_block_cuda -f ./data/weighted/1_nemeth21.el -t 0.6 -b 8
srun nsys profile --stats=true -o ref_block_cuda_thread4 ./ref_block_cuda -f ./data/weighted/1_nemeth21.el -t 0.6 -b 8
# srun ncu  --target-processes all --export ref_block.report --import-source=yes  --metrics regex:sm__inst_executed_pipe_*,regex:sm__sass_thread_inst_executed_op*  --page raw --set full ./ref_block_cuda -f ./data/weighted/1_nemeth21.el -t 0.6 -b 8