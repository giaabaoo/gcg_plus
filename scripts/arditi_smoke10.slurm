#!/bin/bash
#SBATCH --job-name=arditi_smoke10
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=logs/arditi_smoke10_%j.log
#SBATCH --error=logs/arditi_smoke10_%j.err

source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
conda activate codec
cd /home/bao.dinh/Research/gcg_plus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3.12 -c "import torch; print('GPUs:', torch.cuda.device_count())"
python3.12 scripts/run_arditi_smoke10.py
