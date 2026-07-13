#!/bin/bash
#SBATCH --job-name=gcg_arditi_tgt
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --output=logs/gcg_arditi_target_%j.log
#SBATCH --error=logs/gcg_arditi_target_%j.err

source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
conda activate codec
cd /home/bao.dinh/Research/gcg_plus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3.12 scripts/run_gcg_arditi_target.py

python3.12 scripts/score_with_gemma.py \
    --results results/gcg_arditi_target/results.json \
    --out     results/gcg_arditi_target/gemma_scores.json
