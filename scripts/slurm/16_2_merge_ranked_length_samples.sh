#!/bin/bash

#SBATCH -J 16_2_ranked_length_merge
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --oversubscribe
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/projects/small_language_model/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/small_language_model/logs/%x-%j.err

set -euo pipefail

source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate "${SFT_ENV:-sft}"
source /home/youyang7/projects/small_language_model/scripts/slurm/_gpu_idle_gate.sh
cd /home/youyang7/projects/small_language_model

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/mnt/beegfs/youyang7/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/mnt/beegfs/youyang7/.cache/huggingface/datasets}"

CONFIG="${CONFIG:-configs/capacity_length_ranked_sampling_7b_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_ranked_sampling_7b_v1}"
GENERATION_DIR="${OUTPUT_ROOT}/formal/generation"
DATASET_DIR="${OUTPUT_ROOT}/formal/datasets"

python3 -u scripts/16_2_merge_ranked_length_samples.py \
  --config "${CONFIG}" \
  --input-dir "${GENERATION_DIR}" \
  --output-dir "${DATASET_DIR}"

echo "ranked_length_merge_complete output=${DATASET_DIR}"
