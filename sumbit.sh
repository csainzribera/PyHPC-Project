#!/bin/sh
#BSUB -J ex9_gpu
#BSUB -q gpua100             # GPU queue (A100 GPUs)
#BSUB -n 4                    # CPU cores (min 4 required)
#BSUB -W 00:30                # Wall time
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"   # Memory per CPU core
#BSUB -gpu "num=1:mode=exclusive_process"   # Request 1 GPU
#BSUB -R "select[gpu32gb]"   # Optional: choose 40GB GPU (A100)
#BSUB -o %J.out
#BSUB -e %J.err

# Load environment
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

# Load CUDA (required for GPU)
module load cuda

# Run your program
python exercise8.py 50