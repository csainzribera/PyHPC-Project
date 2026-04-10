#!/bin/bash 

#BSUB -J n=50_multiprocessing
#BSUB -q hpc
#BSUB -n 32
#BSUB -W 100
#BSUB -R "rusage[mem=512MB]"
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model == XeonGold6126]"
#BSUB -o n=50_multiprocessing_%J.out
#BSUB -e n=50_multiprocessing_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

python simulate.py 50 1 > n=50_processes=1.txt 
python simulate.py 50 2 > n=50_processes=2.txt
python simulate.py 50 4 > n=50_processes=4.txt
python simulate.py 50 8 > n=50_processes=8.txt
python simulate.py 50 16 > n=50_processes=16.txt
python simulate.py 50 32 > n=50_processes=32.txt