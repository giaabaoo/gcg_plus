#!/bin/bash
#SBATCH --job-name=sample_base
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/sample_base_%j.log
#SBATCH --error=logs/sample_base_%j.err

source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
conda activate gcg_plus
cd /home/bao.dinh/Research/gcg_plus

python scripts/sample_base_interactive.py
