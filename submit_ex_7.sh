#!/bin/sh
#BSUB -J Ex_7
#BSUB -q hpc
#BSUB -n 1
#BSUB -W 00:30
#BSUB -R "rusage[mem=2048]"
#BSUB -o %J.out
#BSUB -e %J.err

# 1. Setup environment
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

# 2. Run the script for 20 floorplans
# We pass '20' as the argument for sys.argv[1]
python simulate_7.py 20