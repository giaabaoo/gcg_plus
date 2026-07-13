#!/bin/bash
#SBATCH --job-name=exp20_gcg_multilayer
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/exp20_%j.log
#SBATCH --error=logs/exp20_%j.log

source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
conda activate codec
cd /home/bao.dinh/Research/gcg_plus

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3.12 -c "import torch; print('GPUs:', torch.cuda.device_count())"

python3.12 scripts/run_exp20_gcg_multilayer_cosine.py

python3.12 scripts/score_with_gemma.py \
    --results results/smoke10_exp20/results.json \
    --out     results/smoke10_exp20/gemma_scores.json
