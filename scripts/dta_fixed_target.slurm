#!/bin/bash
#SBATCH --job-name=dta_fixed
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=logs/dta_fixed_%j.log
#SBATCH --error=logs/dta_fixed_%j.err

source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
conda activate codec
cd /home/bao.dinh/Research/gcg_plus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3.12 -c "import torch; print('GPUs:', torch.cuda.device_count())"

python3.12 scripts/run_dta_fixed_target.py \
    --out results/smoke10_dta_fixed/results.json \
    --local-device cuda:0 \
    --ref-device   cuda:0 \
    --judge-device cuda:0 \
    --num-iters    200 \
    --num-inner    10 \
    --fixed-target "Sure, here is"
