#!/bin/bash

#SBATCH -J 11_0_math_mix_prep
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/projects/small_language_model/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/small_language_model/logs/%x-%j.err

set -euo pipefail
source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate "${PREP_ENV:-fact}"
cd /home/youyang7/projects/small_language_model

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/mnt/beegfs/youyang7/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/mnt/beegfs/youyang7/.cache/huggingface/datasets}"
CONFIG="${CONFIG:-configs/capacity_length_math_mix_pilot_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_math_mix_pilot_v1}"

python3 -u scripts/11_1_prepare_math_mix_source.py \
  --config "${CONFIG}" \
  --output-dir "${OUTPUT_ROOT}/pilot/source"

python3 -u scripts/11_3_prepare_math_mix_eval_suite.py \
  --config "${CONFIG}" \
  --math-train-source "${OUTPUT_ROOT}/pilot/source/math_train_source_1000.jsonl" \
  --output-dir "${OUTPUT_ROOT}/pilot/eval_suite"

echo "math_mix_preparation_complete output=${OUTPUT_ROOT}/pilot"
