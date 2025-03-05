#!/bin/bash
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=72:00:00
#PBS -P gcf51138
#PBS -j oe
#PBS -m n
#PBS -V

# load environmental modules
module load cuda/12.6/12.6.1 cudnn/9.5/9.5.1

# load venv
source ~/venv/AKI/bin/activate

cd $PBS_O_WORKDIR

torchrun --nnodes=1 --nproc_per_node=8 train/instruction_finetune.py