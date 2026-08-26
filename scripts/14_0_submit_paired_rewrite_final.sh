#!/bin/bash
# Submit final paired-rewrite schedules, budget evaluation, analysis, and audit.

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/capacity_length_paired_rewrite_7b_pilot_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_paired_rewrite_7b_pilot_v1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/capacity_length_paired_rewrite_7b_pilot_v1}"
FIGURE_ROOT="${FIGURE_ROOT:-figures/capacity_length_paired_rewrite_7b_pilot_v1}"
UPSTREAM_SELECTION_JOB="${UPSTREAM_SELECTION_JOB:-}"
DRY_RUN="${DRY_RUN:-0}"

if [ "${DRY_RUN}" != "1" ] && [ ! -f "${OUTPUT_ROOT}/pilot/analysis/selection/recipe_selection.json" ] && [ -z "${UPSTREAM_SELECTION_JOB}" ]; then
  echo "Recipe selection is not complete; set UPSTREAM_SELECTION_JOB if it is already submitted." >&2
  exit 2
fi
mkdir -p logs

submit() {
  local dry_id="$1"
  shift
  if [ "${DRY_RUN}" = "1" ]; then
    printf '%q ' "$@" >&2
    printf '\n' >&2
    echo "${dry_id}"
  else
    "$@"
  fi
}

dependency=()
if [ -n "${UPSTREAM_SELECTION_JOB}" ]; then
  dependency=(--dependency="afterok:${UPSTREAM_SELECTION_JOB}")
fi
train_job=$(submit DRY_FINAL_TRAIN \
  sbatch --parsable -p a5000ada -w c32 "${dependency[@]}" \
  --export="ALL,CONFIG=${CONFIG},OUTPUT_ROOT=${OUTPUT_ROOT},CHECKPOINT_ROOT=${CHECKPOINT_ROOT},TRAINING_STAGE=final,SFT_ENV=/mnt/beegfs/youyang7/.conda/envs/sft" \
  scripts/slurm/14_4_train_paired_rewrites.sh | tail -n 1)
eval_job=$(submit DRY_CONFIRMATORY_EVAL \
  sbatch --parsable -p a5000ada -w c32 --dependency="afterok:${train_job}" \
  --export="ALL,CONFIG=${CONFIG},OUTPUT_ROOT=${OUTPUT_ROOT},CHECKPOINT_ROOT=${CHECKPOINT_ROOT},EVAL_STAGE=confirmatory,EVAL_ENV=/mnt/beegfs/youyang7/.conda/envs/sft" \
  scripts/slurm/14_5_eval_paired_rewrites.sh | tail -n 1)
analysis_job=$(submit DRY_FINAL_ANALYSIS \
  sbatch --parsable -p a6000 -w c31 --dependency="afterok:${eval_job}" \
  --export="ALL,CONFIG=${CONFIG},OUTPUT_ROOT=${OUTPUT_ROOT},CHECKPOINT_ROOT=${CHECKPOINT_ROOT},FIGURE_ROOT=${FIGURE_ROOT},ANALYSIS_STAGE=final,ANALYSIS_ENV=/mnt/beegfs/youyang7/.conda/envs/fact" \
  scripts/slurm/14_6_analyze_paired_rewrites.sh | tail -n 1)
audit_job=$(submit DRY_FINAL_AUDIT \
  sbatch --parsable -p a6000 -w c31 --dependency="afterok:${analysis_job}" \
  --export="ALL,CONFIG=${CONFIG},OUTPUT_ROOT=${OUTPUT_ROOT},CHECKPOINT_ROOT=${CHECKPOINT_ROOT},FIGURE_ROOT=${FIGURE_ROOT},AUDIT_ENV=/mnt/beegfs/youyang7/.conda/envs/fact" \
  scripts/slurm/14_7_audit_paired_rewrites.sh | tail -n 1)
echo "final_train_job=${train_job} confirmatory_eval_job=${eval_job} final_analysis_job=${analysis_job} audit_job=${audit_job}"
