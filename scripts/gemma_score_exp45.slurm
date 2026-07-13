#!/bin/bash
#SBATCH --job-name=gemma_exp45
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/gemma_exp45_%j.log
#SBATCH --error=logs/gemma_exp45_%j.err

source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
conda activate gcg_plus
cd /home/bao.dinh/Research/gcg_plus

python scripts/score_with_gemma.py \
    --results results/smoke_rphi_v2/exp4_transfer/results.json \
    --out     results/smoke_rphi_v2/exp4_transfer/gemma_scores.json \
    --batch_size 8

python scripts/score_with_gemma.py \
    --results results/smoke_rphi_v2/exp5_warmstart/results.json \
    --out     results/smoke_rphi_v2/exp5_warmstart/gemma_scores.json \
    --batch_size 8
