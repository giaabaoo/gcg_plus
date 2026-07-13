#!/bin/bash
#SBATCH --job-name=exp13_exp8_250
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=16:00:00
#SBATCH --output=logs/exp13_exp8_250_%j.log
#SBATCH --error=logs/exp13_exp8_250_%j.err

source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
cd /home/bao.dinh/Research/gcg_plus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python main.py \
    --config configs/base.yml \
             configs/models/llama3_8b.yml \
             configs/datasets/smoke_20.yml \
             configs/attacks/gcg_refusal_dir_combined_250.yml
