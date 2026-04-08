#!/bin/bash 

#BSUB -J n=100_multiprocessing
#BSUB -q hpc
#BSUB -n 12
#BSUB -W 100
#BSUB -R "rusage[mem=512MB]"
#BSUB -R "span[hosts=1]"
#BSUB -o n=100_multiprocessing_%J.out
#BSUB -e n=100_multiprocessing_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

python simulate.py 100 1 > n=100_processes=1.txt 
python simulate.py 100 2 > n=100_processes=2.txt
python simulate.py 100 4 > n=100_processes=4.txt
python simulate.py 100 8 > n=100_processes=8.txt
python simulate.py 100 10 > n=100_processes=10.txt
python simulate.py 100 12 > n=100_processes=12.txt