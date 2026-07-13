#!/bin/bash
#SBATCH --job-name=exp16_missing
#SBATCH --partition=long
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=/home/bao.dinh/Research/gcg_plus/logs/exp16_missing_tb_%j.log
#SBATCH --error=/home/bao.dinh/Research/gcg_plus/logs/exp16_missing_tb_%j.err

source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
cd /home/bao.dinh/Research/gcg_plus
export HUGGING_FACE_HUB_TOKEN=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
export HF_TOKEN=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
export HF_HOME=/home/timothy.baldwin/.cache/huggingface
export TRANSFORMERS_CACHE=/home/timothy.baldwin/.cache/huggingface/hub
export HF_HUB_CACHE=/home/timothy.baldwin/.cache/huggingface/hub
export PYTHONPATH=/home/bao.dinh/.local/lib/python3.12/site-packages:$PYTHONPATH

/home/bao.dinh/.local/bin/python main.py \
    --config configs/base.yml \
             configs/models/llama3_8b.yml \
             configs/datasets/advbench_50_missing.yml \
             configs/attacks/gcg_baseline_500.yml
