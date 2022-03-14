#!/usr/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=defq
#SBATCH --time=1-0

source ~/miniconda3/etc/profile.d/conda.sh

conda activate dedalus

date
mpirun -np 24 python3 swimming_sheet_v1.py v1.cfg
date
