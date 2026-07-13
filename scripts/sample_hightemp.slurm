#!/bin/bash
#SBATCH --job-name=base_hightemp
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/base_hightemp_%j.log
#SBATCH --error=logs/base_hightemp_%j.err

source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
cd /home/bao.dinh/Research/gcg_plus

python scripts/sample_hightemp.py

python scripts/score_with_gemma.py \
    --results results/smoke_rphi_v2/base_hightemp/results.json \
    --out     results/smoke_rphi_v2/base_hightemp/gemma_scores.json \
    --batch_size 8
