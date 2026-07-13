#!/bin/bash
#SBATCH --job-name=gemma_exp16
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=logs/gemma_exp16_%j.log
#SBATCH --error=logs/gemma_exp16_%j.err

source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
cd /home/bao.dinh/Research/gcg_plus

python scripts/score_with_gemma.py \
    --results results/advbench_50/exp16_gcg_baseline_500/results.json \
    --out results/advbench_50/exp16_gcg_baseline_500/gemma_scores.json
