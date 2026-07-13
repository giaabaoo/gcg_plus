#!/bin/bash
#SBATCH --job-name=exp15_smoke10
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=logs/exp15_smoke10_%j.log
#SBATCH --error=logs/exp15_smoke10_%j.err

source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
conda activate codec
cd /home/bao.dinh/Research/gcg_plus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3.12 -c "import torch; print('GPUs:', torch.cuda.device_count())"

python3.12 main.py \
    --config configs/base.yml \
             configs/models/llama3_8b.yml \
             configs/datasets/smoke10.yml \
             configs/attacks/exp15_smoke10.yml

python3.12 scripts/score_with_gemma.py \
    --results results/smoke10_exp15/results.json \
    --out     results/smoke10_exp15/gemma_scores.json
