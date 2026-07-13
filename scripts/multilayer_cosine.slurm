#!/bin/bash
#SBATCH --job-name=multilayer_cos
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/multilayer_cos_%j.log
#SBATCH --error=logs/multilayer_cos_%j.err

source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
conda activate codec
cd /home/bao.dinh/Research/gcg_plus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3.12 -c "import torch; print('GPUs:', torch.cuda.device_count())"

python3.12 scripts/run_multilayer_cosine.py --variant both --out_dir results/smoke10_multilayer

python3.12 scripts/score_with_gemma.py \
    --results results/smoke10_multilayer/variant_A.json \
    --out     results/smoke10_multilayer/variant_A_gemma.json

python3.12 scripts/score_with_gemma.py \
    --results results/smoke10_multilayer/variant_B.json \
    --out     results/smoke10_multilayer/variant_B_gemma.json
