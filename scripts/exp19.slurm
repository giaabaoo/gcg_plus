#!/bin/bash
#SBATCH --job-name=exp19_multilayer
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/exp19_%j.log
#SBATCH --error=logs/exp19_%j.err

source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
conda activate codec
cd /home/bao.dinh/Research/gcg_plus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3.12 -c "import torch; print('GPUs:', torch.cuda.device_count())"

python3.12 scripts/run_exp19_multilayer_arditi_target.py --variant both --out_dir results/smoke10_exp19

python3.12 scripts/score_with_gemma.py \
    --results results/smoke10_exp19/variant_A.json \
    --out     results/smoke10_exp19/variant_A_gemma.json

python3.12 scripts/score_with_gemma.py \
    --results results/smoke10_exp19/variant_B.json \
    --out     results/smoke10_exp19/variant_B_gemma.json
