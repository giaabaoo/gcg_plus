#!/bin/bash
#SBATCH --job-name=exp16_missing
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=logs/exp16_missing_%j.log
#SBATCH --error=logs/exp16_missing_%j.err
#SBATCH --nodelist=gpu-05

source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
cd /home/bao.dinh/Research/gcg_plus

python main.py \
    --config configs/base.yml \
             configs/models/llama3_8b.yml \
             configs/datasets/advbench_50_missing.yml \
             configs/attacks/gcg_baseline_500.yml
