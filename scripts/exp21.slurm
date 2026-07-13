#!/bin/bash
#SBATCH --job-name=exp21_gcg_hook
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/exp21_%j.log
#SBATCH --error=logs/exp21_%j.err

source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
conda activate codec
cd /home/bao.dinh/Research/gcg_plus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1

python3.12 -c "import torch; print('GPUs:', torch.cuda.device_count())"

python3.12 scripts/run_exp21_gcg_arditi_hook_target.py

python3.12 scripts/score_with_gemma.py \
    --results results/smoke10_exp21/results.json \
    --out     results/smoke10_exp21/gemma_scores.json
