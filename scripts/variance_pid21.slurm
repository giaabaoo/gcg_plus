#!/bin/bash
#SBATCH --job-name=var_pid21
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=logs/var_pid21_%A_%a.log
#SBATCH --error=logs/var_pid21_%A_%a.err
#SBATCH --array=1-3

echo "Variance pid21 | Job $SLURM_JOB_ID | Array $SLURM_ARRAY_TASK_ID | Node $SLURMD_NODENAME | Start $(date)"
source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
cd /home/bao.dinh/Research/gcg_plus
nvidia-smi | head -5

SEED=${SLURM_ARRAY_TASK_ID}

python main.py \
    --config configs/base.yml \
             configs/models/llama3_8b.yml \
             configs/datasets/strongreject_pid21.yml \
             configs/attacks/gcg.yml \
    --attack.seed ${SEED}:int \
    --exp_name pid21_seed${SEED}

if [ $? -ne 0 ]; then echo "ERROR: variance run failed!"; exit 1; fi
echo "Done. $(date)"
