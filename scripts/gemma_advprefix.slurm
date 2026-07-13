#!/bin/bash
#SBATCH --job-name=gemma_advpfx
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=logs/gemma_advprefix_%j.log
#SBATCH --error=logs/gemma_advprefix_%j.err

source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
cd /home/bao.dinh/Research/gcg_plus

python scripts/score_with_gemma.py \
    --results results/advbench_50/smoke10_advprefix_gcg/shard0.json \
              results/advbench_50/smoke10_advprefix_gcg/shard1.json \
    --out results/advbench_50/smoke10_advprefix_gcg/gemma_scores.json
