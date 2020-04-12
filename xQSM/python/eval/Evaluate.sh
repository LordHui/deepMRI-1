#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=Demo
#SBATCH -n 1
#SBATCH -c 3
#SBATCH --mem=50000
#SBATCH -e Error.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla-smx2:1

module load anaconda/3.6
source activate /opt/ohpc/pub/apps/pytorch_1.10_openmpi
module load cuda/10.0.130
module load gnu/5.4.0
module load mvapich2
module load matlab

srun python -u run_demo.py
