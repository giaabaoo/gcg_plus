#!/bin/bash
#SBATCH --job-name=gemma_exp8_10
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/gemma_exp8_10_%j.log
#SBATCH --error=logs/gemma_exp8_10_%j.err

source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
cd /home/bao.dinh/Research/gcg_plus

python scripts/score_with_gemma.py \
    --results results/smoke_rphi_v2/exp8_refusal_dir_combined/results.json \
    --out     results/smoke_rphi_v2/exp8_refusal_dir_combined/gemma_scores.json \
    --batch_size 8

python scripts/score_with_gemma.py \
    --results results/smoke_rphi_v2/exp10_gcg_rphi_verify/results.json \
    --out     results/smoke_rphi_v2/exp10_gcg_rphi_verify/gemma_scores.json \
    --batch_size 8
