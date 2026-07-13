#!/bin/bash
#SBATCH --job-name=univ_arditi
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=logs/univ_arditi_%j.log
#SBATCH --error=logs/univ_arditi_%j.err

source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
conda activate codec
cd /home/bao.dinh/Research/gcg_plus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3.12 -c "import torch; print('GPUs:', torch.cuda.device_count())"

python3.12 scripts/run_universal_arditi_gcg.py

python3.12 scripts/score_with_gemma.py \
    --results results/smoke10_universal_arditi/results.json \
    --out     results/smoke10_universal_arditi/gemma_scores.json
