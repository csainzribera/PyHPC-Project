#BSUB -J ex12_gpu
#BSUB -q gpuv100 # GPU queue (A100 GPUs)
#BSUB -n 8                 # CPU cores (min 4 required)
#BSUB -W 04:30                # Wall time
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=5GB]"   # Memory per CPU core
#BSUB -gpu "num=2:mode=exclusive_process"   # Request 2 GPUs
#BSUB -R "select[gpu32gb]"   # Optional: choose 40GB GPU (A100)
#BSUB -o ex_%J.out
#BSUB -e ex_%J.err

# Load environment
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

# Load CUDA (required for GPU)
module load cuda/11.8

# 3. Run the script
# Processing 50 floorplans with 20,000 iterations on a GPU 
# should easily fit within your 30-minute window.
python ex12new.py 4500
