#!/bin/bash
#SBATCH --job-name=gemma_exp17
#SBATCH --partition=long
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/home/bao.dinh/Research/gcg_plus/logs/gemma_exp17_tb_%j.log
#SBATCH --error=/home/bao.dinh/Research/gcg_plus/logs/gemma_exp17_tb_%j.err

source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
cd /home/bao.dinh/Research/gcg_plus
export HUGGING_FACE_HUB_TOKEN=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
export HF_TOKEN=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
export HF_HOME=/home/timothy.baldwin/.cache/huggingface
export TRANSFORMERS_CACHE=/home/timothy.baldwin/.cache/huggingface/hub
export HF_HUB_CACHE=/home/timothy.baldwin/.cache/huggingface/hub
export PYTHONPATH=/home/bao.dinh/.local/lib/python3.12/site-packages:$PYTHONPATH

/home/bao.dinh/.local/bin/python scripts/score_with_gemma.py \
    --results results/advbench_50/exp17_cefloor_500steps/results.json \
    --out  results/advbench_50/exp17_cefloor_500steps/gemma_scores.json
