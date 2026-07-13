#!/bin/bash
#SBATCH --job-name=train_rphi_v2
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/train_rphi_v2_%j.log
#SBATCH --error=logs/train_rphi_v2_%j.err

echo "Train rphi v2 | Job $SLURM_JOB_ID | Node $SLURMD_NODENAME | Start $(date)"
source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
cd /home/bao.dinh/Research/gcg_plus
nvidia-smi | head -5

# Step 3: Build combined train/val set
echo "=== Building trainset v2 ==="
python scripts/build_rphi_trainset_v2.py \
    --v2_pos_pool   data/rphi_pool_v2/naive_pos_2600.json \
    --v2_neg_pool   data/rphi_pool_v2/naive_neg_734.json \
    --v2_pos_scores results/harmbench/scores/rphi_v2_pos_gemma.json \
    --v1_pos_pool   data/rphi_pool/naive_pos_500.json \
    --v1_neg_pool   data/rphi_pool/naive_neg_200.json \
    --v1_pos_scores results/harmbench/scores/rphi_pos_gemma.json \
    --out_dir       data/rphi_train_v2

if [ $? -ne 0 ]; then echo "ERROR: build_rphi_trainset_v2 failed"; exit 1; fi

# Step 4: Train DeBERTa v2
echo "=== Training rphi v2 ==="
python scripts/train_rphi_v2.py \
    --train_path data/rphi_train_v2/train.jsonl \
    --val_path   data/rphi_train_v2/val.jsonl \
    --out_dir    checkpoints/rphi_v2 \
    --epochs     5 \
    --batch_size 16 \
    --lr         2e-5

if [ $? -ne 0 ]; then echo "ERROR: train_rphi_v2 failed"; exit 1; fi

echo "Done. $(date)"
