#!/bin/bash
#SBATCH --job-name=exp14_gcg_250
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=logs/exp14_gcg_baseline_250_%j.log
#SBATCH --error=logs/exp14_gcg_baseline_250_%j.err

source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
cd /home/bao.dinh/Research/gcg_plus

python main.py \
    --config configs/base.yml \
             configs/models/llama3_8b.yml \
             configs/datasets/smoke_20.yml \
             configs/attacks/gcg_baseline_250.yml
