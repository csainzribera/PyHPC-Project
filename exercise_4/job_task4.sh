#!/bin/bash

#BSUB -J projectTask4
#BSUB -q hpc
#BSUB -W 30
#BSUB -R "rusage[mem=512MB]"
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -o logFiles/jobTask4_%J.out
#BSUB -e logFiles/jobTask4_%J.err

#BSUB -R "select[model == XeonGold6126]"

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

kernprof -l task4.py 10
python -m line_profiler -rmt task4.py.lprof