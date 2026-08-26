#!/bin/bash

#SBATCH -J 13_3_freeze_kd
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/home/%u/projects/small_language_model/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/small_language_model/logs/%x-%j.err

set -euo pipefail
source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate "${PREP_ENV:-fact}"
cd /home/youyang7/projects/small_language_model

CONFIG="${CONFIG:-configs/capacity_length_multibench_multiteacher_kd_pilot_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_multibench_multiteacher_kd_pilot_v1}"
python3 -u scripts/13_3_freeze_multiteacher_kd_protocol.py \
  --config "${CONFIG}" \
  --dataset-manifest "${OUTPUT_ROOT}/pilot/sft_data/dataset_manifest.json" \
  --output-json "${OUTPUT_ROOT}/pilot/frozen_kd_protocol.json"
