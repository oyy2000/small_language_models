#!/bin/bash

#SBATCH -J 14_6_paired_rewrite_analysis
#SBATCH -p a6000
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/projects/small_language_model/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/small_language_model/logs/%x-%j.err

set -euo pipefail
source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate "${ANALYSIS_ENV:-fact}"
cd /home/youyang7/projects/small_language_model

CONFIG="${CONFIG:-configs/capacity_length_paired_rewrite_7b_pilot_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_paired_rewrite_7b_pilot_v1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/capacity_length_paired_rewrite_7b_pilot_v1}"
FIGURE_ROOT="${FIGURE_ROOT:-figures/capacity_length_paired_rewrite_7b_pilot_v1}"
ANALYSIS_STAGE="${ANALYSIS_STAGE:-selection}"

mkdir -p "${FIGURE_ROOT}/${ANALYSIS_STAGE}"
if [ "${ANALYSIS_STAGE}" = "selection" ]; then
  python3 -u scripts/14_6_analyze_paired_rewrite_pilot.py \
    --config "${CONFIG}" \
    --mode selection \
    --eval-manifest-glob "${OUTPUT_ROOT}/pilot/eval/selection/eval_manifest_*.json" \
    --training-manifest-glob "${CHECKPOINT_ROOT}/grid/*/training_manifest.json" \
    --dataset-manifest "${OUTPUT_ROOT}/pilot/sft_data/dataset_manifest.json" \
    --output-dir "${OUTPUT_ROOT}/pilot/analysis/selection" \
    --figure-dir "${FIGURE_ROOT}/selection"
elif [ "${ANALYSIS_STAGE}" = "final" ]; then
  python3 -u scripts/14_6_analyze_paired_rewrite_pilot.py \
    --config "${CONFIG}" \
    --mode final \
    --eval-manifest-glob "${OUTPUT_ROOT}/pilot/eval/confirmatory/eval_manifest_*.json" \
    --selection-json "${OUTPUT_ROOT}/pilot/analysis/selection/recipe_selection.json" \
    --output-dir "${OUTPUT_ROOT}/pilot/analysis/final" \
    --figure-dir "${FIGURE_ROOT}/final"
else
  echo "Unsupported ANALYSIS_STAGE=${ANALYSIS_STAGE}" >&2
  exit 2
fi
