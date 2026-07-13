#!/bin/bash
# Score a results.json with the Gemma StrongREJECT judge.
#
# Usage:
#   sbatch scripts/score.slurm results/gcg_full_budget/results.json \
#          results/gcg_full_budget/gemma_scores.json
#
#SBATCH --job-name=gemma_score
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
conda activate codec
cd /home/bao.dinh/Research/gcg_plus

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RESULTS="${1:?usage: sbatch scripts/score.slurm <results.json> <out.json>}"
OUT="${2:?usage: sbatch scripts/score.slurm <results.json> <out.json>}"

python3.12 scripts/score_with_gemma.py --results "$RESULTS" --out "$OUT"
