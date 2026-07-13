#!/bin/bash
#SBATCH --job-name=advpfx_gcg_s1
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=logs/advprefix_gcg_shard1_%j.log
#SBATCH --error=logs/advprefix_gcg_shard1_%j.err

source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
cd /home/bao.dinh/Research/gcg_plus

python scripts/run_advprefix_gcg.py \
    --shard 1 --num_shards 2 \
    --num_steps 250 \
    --out results/advbench_50/smoke10_advprefix_gcg/shard1.json
