#!/bin/bash

#SBATCH -J 11_2_math_mix_build
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/projects/small_language_model/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/small_language_model/logs/%x-%j.err

set -euo pipefail
source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate "${BUILD_ENV:-fact}"
cd /home/youyang7/projects/small_language_model

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/mnt/beegfs/youyang7/.cache/huggingface}"
CONFIG="${CONFIG:-configs/capacity_length_math_mix_pilot_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_math_mix_pilot_v1}"

python3 -u scripts/5_2_merge_select_capacity_length_traces.py \
  --config "${CONFIG}" \
  --input-glob "${OUTPUT_ROOT}/pilot/raw/qwen2p5_7b/shard_*.jsonl" \
  --output-dir "${OUTPUT_ROOT}/pilot/selected" \
  --stage pilot \
  --expected-problems 1000

python3 -u scripts/11_2_build_math_mix_sft_data.py \
  --config "${CONFIG}" \
  --math-selected-traces "${OUTPUT_ROOT}/pilot/selected/selected_traces.jsonl" \
  --math-selection-audit "${OUTPUT_ROOT}/pilot/selected/selection_audit.json" \
  --output-dir "${OUTPUT_ROOT}/pilot/sft_data"

echo "math_mix_build_complete output=${OUTPUT_ROOT}/pilot/sft_data"
