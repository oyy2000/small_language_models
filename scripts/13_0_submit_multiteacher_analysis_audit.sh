#!/bin/bash
# Submit analysis and completion audit after all evaluation launcher shards pass.

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/capacity_length_multibench_multiteacher_kd_pilot_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_multibench_multiteacher_kd_pilot_v1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/capacity_length_multibench_multiteacher_kd_pilot_v1/pilot}"
FIGURE_ROOT="${FIGURE_ROOT:-figures/capacity_length_multibench_multiteacher_kd_pilot_v1}"
EVAL_JOB_IDS="${EVAL_JOB_IDS:?EVAL_JOB_IDS is required as a colon-separated list}"
DRY_RUN="${DRY_RUN:-0}"

if [ "${DRY_RUN}" = "1" ]; then
  echo "sbatch --dependency=afterok:${EVAL_JOB_IDS} scripts/slurm/13_8_analyze_multiteacher_multibench_kd.sh"
  echo "sbatch --dependency=afterok:DRY_ANALYSIS scripts/slurm/13_9_audit_multiteacher_multibench_kd.sh"
  exit 0
fi

analysis_job=$(sbatch --parsable -p a6000 -w c31 \
  --dependency="afterok:${EVAL_JOB_IDS}" \
  --export="ALL,OUTPUT_ROOT=${OUTPUT_ROOT},FIGURE_ROOT=${FIGURE_ROOT},ANALYSIS_ENV=/mnt/beegfs/youyang7/.conda/envs/fact" \
  scripts/slurm/13_8_analyze_multiteacher_multibench_kd.sh)
audit_job=$(sbatch --parsable -p a6000 -w c31 \
  --dependency="afterok:${analysis_job}" \
  --export="ALL,CONFIG=${CONFIG},OUTPUT_ROOT=${OUTPUT_ROOT},CHECKPOINT_ROOT=${CHECKPOINT_ROOT},FIGURE_ROOT=${FIGURE_ROOT},AUDIT_ENV=/mnt/beegfs/youyang7/.conda/envs/fact" \
  scripts/slurm/13_9_audit_multiteacher_multibench_kd.sh)
echo "analysis_job=${analysis_job} audit_job=${audit_job}"
