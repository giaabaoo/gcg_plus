#!/bin/bash
#SBATCH --job-name=dta_smoke10_v2
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=logs/dta_smoke10_v2_%j.log
#SBATCH --error=logs/dta_smoke10_v2_%j.err

source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
conda activate codec
cd /home/bao.dinh/Research/gcg_plus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python3.12 -c "import torch; n=torch.cuda.device_count(); print('GPUs:', n); assert n>=2, f'Need 2 GPUs, got {n}'"

python3.12 scripts/run_dta_smoke10.py \
    --out results/smoke10_dta_v2/results.json \
    --local-device cuda:0 \
    --ref-device   cuda:1 \
    --judge-device cuda:1 \
    --num-iters    200 \
    --num-inner    10 \
    --sample-count 30
