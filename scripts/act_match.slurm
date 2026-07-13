#!/bin/bash
#SBATCH --job-name=act_match
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/act_match_%j.log
#SBATCH --error=logs/act_match_%j.err

source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
conda activate codec
cd /home/bao.dinh/Research/gcg_plus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3.12 scripts/run_act_match.py

python3.12 scripts/score_with_gemma.py \
    --results results/act_match/results.json \
    --out     results/act_match/gemma_scores.json
