#!/bin/bash
#SBATCH --job-name=exp17_cf_s1b
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=16:00:00
#SBATCH --output=logs/exp17_cefloor_500_s1_%j.log
#SBATCH --error=logs/exp17_cefloor_500_s1_%j.err

source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
cd /home/bao.dinh/Research/gcg_plus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python main.py \
    --config configs/base.yml \
             configs/models/llama3_8b.yml \
             configs/datasets/advbench_50_s1.yml \
             configs/attacks/gcg_refusal_dir_combined_500_cefloor.yml
