#!/bin/bash
#SBATCH --job-name=exp15_cefloor
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=16:00:00
#SBATCH --output=logs/exp15_cefloor_%j.log
#SBATCH --error=logs/exp15_cefloor_%j.err

source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
cd /home/bao.dinh/Research/gcg_plus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python main.py \
    --config configs/base.yml \
             configs/models/llama3_8b.yml \
             configs/datasets/smoke_20.yml \
             configs/attacks/gcg_refusal_dir_combined_250_cefloor.yml

python scripts/score_with_gemma.py \
    --results results/full_20/exp15_cefloor_250steps/results.json \
    --out     results/full_20/exp15_cefloor_250steps/gemma_scores.json \
    --batch_size 8
