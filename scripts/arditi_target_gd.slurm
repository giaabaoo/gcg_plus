#!/bin/bash
#SBATCH --job-name=arditi_tgt_gd
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/arditi_tgt_gd_%j.log
#SBATCH --error=logs/arditi_tgt_gd_%j.err

source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
conda activate codec
cd /home/bao.dinh/Research/gcg_plus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3.12 -c "import torch; print('GPUs:', torch.cuda.device_count())"

python3.12 scripts/run_arditi_target_gd.py

python3.12 scripts/score_with_gemma.py \
    --results results/smoke10_arditi_target_gd/results.json \
    --out     results/smoke10_arditi_target_gd/gemma_scores.json
