#!/bin/bash

#SBATCH -J 13_9_multiteacher_audit
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/projects/small_language_model/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/small_language_model/logs/%x-%j.err

set -euo pipefail
source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate "${AUDIT_ENV:-fact}"
cd /home/youyang7/projects/small_language_model

CONFIG="${CONFIG:-configs/capacity_length_multibench_multiteacher_kd_pilot_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_multibench_multiteacher_kd_pilot_v1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/capacity_length_multibench_multiteacher_kd_pilot_v1/pilot}"
FIGURE_ROOT="${FIGURE_ROOT:-figures/capacity_length_multibench_multiteacher_kd_pilot_v1}"
python3 -u scripts/13_9_audit_multiteacher_multibench_kd.py \
  --config "${CONFIG}" \
  --output-root "${OUTPUT_ROOT}" \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --figure-root "${FIGURE_ROOT}" \
  --output-json "${OUTPUT_ROOT}/pilot/completion_audit.json"
