#!/bin/bash
#SBATCH --job-name=rphi_pool_v2
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/rphi_pool_v2_%j.log
#SBATCH --error=logs/rphi_pool_v2_%j.err

echo "Build+Score rphi pool v2 | Job $SLURM_JOB_ID | Node $SLURMD_NODENAME | Start $(date)"
source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
cd /home/bao.dinh/Research/gcg_plus
nvidia-smi | head -5

# Step 1: Build pool (CPU-only, fast)
echo "=== Building pool v2 ==="
python scripts/build_rphi_pool_v2.py \
    --naive_pos 2600 \
    --naive_neg 1186 \
    --seed 123 \
    --out_dir data/rphi_pool_v2

if [ $? -ne 0 ]; then echo "ERROR: build_rphi_pool_v2 failed"; exit 1; fi

# Step 2: Score naive-positives with Gemma judge
echo "=== Scoring with Gemma ==="
mkdir -p results/harmbench/scores

python scripts/score_with_gemma.py \
    --results data/rphi_pool_v2/naive_pos_2600.json \
    --out     results/harmbench/scores/rphi_v2_pos_gemma.json \
    --batch_size 16

if [ $? -ne 0 ]; then echo "ERROR: score_with_gemma failed"; exit 1; fi

echo "Done. $(date)"
