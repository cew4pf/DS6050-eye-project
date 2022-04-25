#!/bin/bash
#SBATCH -J dlproj
#SBATCH -o dlproj-%A.out
#SBATCH -e dlproj-%A.err
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --time=1-00:00:00

module purge
module load singularity
module load tensorflow/2.8.0
module load anaconda

singularity run --nv /home/cew4pf/tensorflow_2.8.0.sif $1
