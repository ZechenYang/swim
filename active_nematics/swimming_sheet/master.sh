#!/usr/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh

conda activate dedalus

# sbatch run_swimming.sh

# python3 plot_data.py snapshots_test/snapshots_test_s*.h5

# python3 -m dedalus merge_procs snapshots_test/snapshots_test_s1/

# squeue

# python3 plot_data.py output_data/output_v1/snapshots/snapshots_s*.h5

mpirun -np 13 python3 plot_data.py output_data/output_v1/snapshots/snapshots_s*.h5
