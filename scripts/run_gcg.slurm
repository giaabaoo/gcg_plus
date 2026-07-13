#!/bin/bash
#SBATCH --job-name=gcg_plus
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/gcg_%A_%a.log
#SBATCH --error=logs/gcg_%A_%a.err

# Usage:
#   Single run:   sbatch run_gcg.slurm <dataset>
#   Parallel:     sbatch --array=0-2 run_gcg.slurm <dataset> 3
#
# Args:
#   $1  dataset name: strongreject | harmbench  (default: strongreject)
#   $2  num_shards: total parallel jobs         (default: 1)

echo "GCG-Plus | Job $SLURM_JOB_ID | Array $SLURM_ARRAY_TASK_ID | Node $SLURMD_NODENAME | Start $(date)"
source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
cd /home/bao.dinh/Research/gcg_plus
nvidia-smi | head -20

DATASET=${1:-strongreject}
NUM_SHARDS=${2:-1}
SHARD_IDX=${SLURM_ARRAY_TASK_ID:-0}

echo "=== GCG on $DATASET | shard $SHARD_IDX / $NUM_SHARDS ==="
python main.py \
    --config configs/base.yml \
             configs/models/llama3_8b.yml \
             configs/datasets/${DATASET}.yml \
             configs/attacks/gcg.yml \
    --dataset.num_shards ${NUM_SHARDS}:int \
    --dataset.shard_idx  ${SHARD_IDX}:int \
    --exp_name gcg_shard${SHARD_IDX}of${NUM_SHARDS}

if [ $? -ne 0 ]; then echo "ERROR: GCG run failed!"; exit 1; fi
echo "Done. $(date)"
