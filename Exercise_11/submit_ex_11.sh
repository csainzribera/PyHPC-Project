#!/bin/sh
#BSUB -J Ex_11
#BSUB -q hpc
#BSUB -n 24                          # número de cores (= n_workers)
#BSUB -W 00:30
#BSUB -R "span[hosts=1]"             # IMPORTANTE: todos los cores en 1 nodo
#BSUB -R "rusage[mem=4096]"          # memoria POR core (4GB × 16 = 64GB total)
#BSUB -R "select[model == XeonGold6126]"
#BSUB -o %J.out
#BSUB -e %J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

# $LSB_DJOB_NUMPROC contiene automáticamente el nº de cores asignados por LSF
python simulate_11.py 50 24