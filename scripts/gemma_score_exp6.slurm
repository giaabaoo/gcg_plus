#!/bin/bash
#SBATCH --job-name=gemma_exp6
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/gemma_exp6_%j.log
#SBATCH --error=logs/gemma_exp6_%j.err

source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
conda activate gcg_plus
cd /home/bao.dinh/Research/gcg_plus

python scripts/score_with_gemma.py \
    --results results/smoke_rphi_v2/exp6_base_target/results.json \
    --out     results/smoke_rphi_v2/exp6_base_target/gemma_scores.json \
    --batch_size 8
