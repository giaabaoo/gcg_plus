#!/bin/bash
#SBATCH --job-name=dta_gpu_sanity
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:15:00
#SBATCH --output=logs/dta_gpu_sanity_%j.log
#SBATCH --error=logs/dta_gpu_sanity_%j.err

source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
conda activate codec
cd /home/bao.dinh/Research/gcg_plus

python3.12 scripts/dta_sanity_check.py
