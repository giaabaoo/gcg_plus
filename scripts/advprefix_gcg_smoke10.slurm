#!/bin/bash
#SBATCH --job-name=advprefix_gcg
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/advprefix_gcg_smoke10_%j.log
#SBATCH --error=logs/advprefix_gcg_smoke10_%j.err

source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
cd /home/bao.dinh/Research/gcg_plus

python scripts/run_advprefix_gcg.py \
    --out results/advbench_50/smoke10_advprefix_gcg/results.json
