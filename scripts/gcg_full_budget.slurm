#!/bin/bash
#SBATCH --job-name=gcg_full_budget
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=logs/gcg_full_budget_%j.log
#SBATCH --error=logs/gcg_full_budget_%j.err

source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
conda activate codec
cd /home/bao.dinh/Research/gcg_plus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3.12 scripts/run_gcg_full_budget.py

python3.12 scripts/score_with_gemma.py \
    --results results/gcg_full_budget/results.json \
    --out     results/gcg_full_budget/gemma_scores.json
