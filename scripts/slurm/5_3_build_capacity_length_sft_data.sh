#!/bin/bash

#SBATCH -J 5_3_capacity_length_data
#SBATCH -p a6000
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --oversubscribe
#SBATCH --time=08:00:00
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

CONFIG="${CONFIG:-configs/capacity_length_factorial_v1.json}"
STAGE="${STAGE:-smoke}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_factorial_v1}"
STAGE_ROOT="${OUTPUT_ROOT}/${STAGE}"

test -f "${STAGE_ROOT}/selected/SELECTION_COMPLETE"
mkdir_with_retry "${STAGE_ROOT}/sft_data/equal_example"
mkdir_with_retry "${STAGE_ROOT}/sft_data/equal_token"
mkdir_with_retry "${STAGE_ROOT}/sft_data/calibration"

python3 -u scripts/5_3_build_capacity_length_sft_data.py \
  --config "${CONFIG}" \
  --selected-traces "${STAGE_ROOT}/selected/selected_traces.jsonl" \
  --output-dir "${STAGE_ROOT}/sft_data" \
  --stage "${STAGE}"

echo "dataset_build_complete stage=${STAGE} output=${STAGE_ROOT}/sft_data"
