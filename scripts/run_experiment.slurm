#!/bin/bash
# Generic GPU job: run any python entry point in this repo.
#
# Usage:
#   sbatch scripts/run_experiment.slurm scripts/run_gcg_full_budget.py
#   sbatch scripts/run_experiment.slurm main.py --config configs/base.yml \
#          configs/models/llama3_8b.yml configs/datasets/advbench_50.yml \
#          configs/attacks/gcg.yml
#
#SBATCH --job-name=gcg
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
conda activate codec
cd /home/bao.dinh/Research/gcg_plus

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NOTE: conda's own python is 3.14 (no torch). torch/transformers/nanogcg live in
# the python3.12 user site-packages, so ALWAYS invoke python3.12 explicitly.
python3.12 "$@"
