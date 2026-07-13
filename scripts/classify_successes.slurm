#!/bin/bash
#SBATCH --job-name=classify_succ
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/classify_successes_%j.log
#SBATCH --error=logs/classify_successes_%j.err

source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
cd /home/bao.dinh/Research/gcg_plus

python scripts/classify_successes.py \
    --out results/full_20/type1_type2_classification.json
