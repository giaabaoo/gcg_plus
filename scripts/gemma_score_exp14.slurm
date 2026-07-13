#!/bin/bash
#SBATCH --job-name=gemma_exp14
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/gemma_exp14_%j.log
#SBATCH --error=logs/gemma_exp14_%j.err

source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
cd /home/bao.dinh/Research/gcg_plus

python scripts/score_with_gemma.py \
    --results results/full_20/exp14_gcg_baseline_250/results.json \
    --out     results/full_20/exp14_gcg_baseline_250/gemma_scores.json \
    --batch_size 8
