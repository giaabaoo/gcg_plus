#!/bin/bash
#SBATCH --job-name=eval_no_hook
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_no_hook_%j.log
#SBATCH --error=logs/eval_no_hook_%j.err

source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
conda activate codec
cd /home/bao.dinh/Research/gcg_plus

python3.12 scripts/eval_suffix_no_hook.py

python3.12 scripts/score_with_gemma.py \
    --results results/smoke10_exp21/results_no_hook.json \
    --out     results/smoke10_exp21/gemma_scores_no_hook.json
