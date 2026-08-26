#!/bin/bash

#SBATCH -J 14_2_paired_rewrite_build
#SBATCH -p a6000
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/projects/small_language_model/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/small_language_model/logs/%x-%j.err

set -euo pipefail
source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate "${BUILD_ENV:-sft}"
cd /home/youyang7/projects/small_language_model

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/mnt/beegfs/youyang7/.cache/huggingface}"

CONFIG="${CONFIG:-configs/capacity_length_paired_rewrite_7b_pilot_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_paired_rewrite_7b_pilot_v1}"
python3 -u scripts/14_2_build_paired_rewrite_sft_data.py \
  --config "${CONFIG}" \
  --rewrite-glob "${OUTPUT_ROOT}/pilot/raw/shard_*.jsonl" \
  --output-dir "${OUTPUT_ROOT}/pilot/sft_data"
