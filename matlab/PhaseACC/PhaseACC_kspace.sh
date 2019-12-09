#!/bin/bash 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=Yang_test_GPU
#SBATCH -c 3
#SBATCH --mem=40000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla-smx2:2

module purge
module load gnu7
module load cuda/10.0.130
module load anaconda/3.6
module load mvapich2
module load matlab/R2019b

srun matlab -r "TrainCConvNet130_mse_kspace"
