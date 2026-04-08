#!/bin/bash 

#BSUB -J first_run
#BSUB -q hpc
#BSUB -W 60
#BSUB -R "rusage[mem=512MB]"
#BSUB -R "span[hosts=1]"
#BSUB -o first_run_%J.out
#BSUB -e first_run_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

python simulate.py 10




