#!/bin/bash

#SBATCH -J 8_1_capacity_length_audit
#SBATCH -p a6000
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/projects/small_language_model/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/small_language_model/logs/%x-%j.err

set -euo pipefail

source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate "${SFT_ENV:-sft}"
cd /home/youyang7/projects/small_language_model

CONFIG="${CONFIG:-configs/capacity_length_factorial_v1.json}"
STAGE="${STAGE:-formal}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_factorial_v1}"
STAGE_ROOT="${OUTPUT_ROOT}/${STAGE}"

python3 -u scripts/8_1_audit_capacity_length_completion.py \
  --config "${CONFIG}" \
  --stage "${STAGE}" \
  --stage-root "${STAGE_ROOT}" \
  --output-json "${STAGE_ROOT}/completion_audit.json"
