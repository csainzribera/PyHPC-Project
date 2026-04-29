#!/bin/bash

#BSUB -J projectTask2
#BSUB -q hpc
#BSUB -W 30
#BSUB -R "rusage[mem=1GB]"
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -o logFiles/jobTask2_%J.out
#BSUB -e logFiles/jobTask2_%J.err

#BSUB -R "select[model == XeonGold6126]"

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

time python task2.py 20