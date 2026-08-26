#!/bin/bash

#SBATCH -J 13_8_multiteacher_analysis
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/projects/small_language_model/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/small_language_model/logs/%x-%j.err

set -euo pipefail
source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate "${ANALYSIS_ENV:-fact}"
cd /home/youyang7/projects/small_language_model

OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_multibench_multiteacher_kd_pilot_v1}"
FIGURE_ROOT="${FIGURE_ROOT:-figures/capacity_length_multibench_multiteacher_kd_pilot_v1}"
mkdir -p "${FIGURE_ROOT}"
python3 -u scripts/13_8_analyze_multiteacher_multibench_kd.py \
  --protocol "${OUTPUT_ROOT}/pilot/frozen_kd_protocol.json" \
  --eval-dir "${OUTPUT_ROOT}/pilot/eval" \
  --output-dir "${OUTPUT_ROOT}/pilot/analysis" \
  --figure-dir "${FIGURE_ROOT}" \
  --bootstrap-samples 10000
