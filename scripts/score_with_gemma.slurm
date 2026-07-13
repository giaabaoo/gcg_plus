#!/bin/bash
#SBATCH --job-name=gemma_judge
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/gemma_judge_%j.log
#SBATCH --error=logs/gemma_judge_%j.err

# Usage:
#   sbatch scripts/score_with_gemma.slurm

echo "Gemma judge | Job $SLURM_JOB_ID | Node $SLURMD_NODENAME | Start $(date)"
source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
cd /home/bao.dinh/Research/gcg_plus
nvidia-smi | head -20

python scripts/score_with_gemma.py \
    --results data/rphi_pool/naive_pos_500.json \
    --out     results/harmbench/scores/rphi_pos_gemma.json

if [ $? -ne 0 ]; then echo "ERROR: judge run failed!"; exit 1; fi
echo "Done. $(date)"
