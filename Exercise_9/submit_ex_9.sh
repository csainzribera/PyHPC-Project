#!/bin/sh
#BSUB -q c02613
#BSUB -J simulate_9
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:30
#BSUB -o %J.out
#BSUB -e %J.err

# 1. Setup environment
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026    

# 2. Run the script for 50 floorplans
python simulate_9.py 50