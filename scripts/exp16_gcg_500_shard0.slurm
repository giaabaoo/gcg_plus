#!/bin/bash
#SBATCH --job-name=exp16_gcg_s0
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=16:00:00
#SBATCH --output=logs/exp16_gcg_500_s0_%j.log
#SBATCH --error=logs/exp16_gcg_500_s0_%j.err

source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
cd /home/bao.dinh/Research/gcg_plus

python main.py \
    --config configs/base.yml \
             configs/models/llama3_8b.yml \
             configs/datasets/advbench_50_s0.yml \
             configs/attacks/gcg_baseline_500.yml
