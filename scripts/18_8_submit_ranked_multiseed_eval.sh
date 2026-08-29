#!/bin/bash
# Submit Phase-A evaluation after the six-adapter training job succeeds.

set -euo pipefail
cd "$(dirname "$0")/.."

TRAIN_JOB_ID="${TRAIN_JOB_ID:-276988}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_ranked_sampling_7b_multiseed_v1}"
NODE="${NODE:-c32}"
PARTITION="${PARTITION:-a5000ada}"

if [ -e "${OUTPUT_ROOT}/MULTISEED_COMPLETE" ]; then
  echo "Multi-seed experiment is already sealed: ${OUTPUT_ROOT}/MULTISEED_COMPLETE" >&2
  exit 2
fi
if ! squeue -h -j "${TRAIN_JOB_ID}" >/dev/null 2>&1; then
  state=$(sacct -j "${TRAIN_JOB_ID}" --format=State -n -X | awk 'NF {print $1; exit}')
  if [ "${state}" != "COMPLETED" ]; then
    echo "Training job ${TRAIN_JOB_ID} is neither live nor completed: state=${state:-unknown}" >&2
    exit 2
  fi
fi

job_id=$(sbatch --parsable \
  -p "${PARTITION}" \
  -w "${NODE}" \
  --dependency="afterok:${TRAIN_JOB_ID}" \
  --export="ALL,OUTPUT_ROOT=${OUTPUT_ROOT}" \
  scripts/slurm/18_5_eval_analyze_audit_ranked_multiseed.sh)
echo "training_job=${TRAIN_JOB_ID} evaluation_job=${job_id} dependency=afterok:${TRAIN_JOB_ID} node=${NODE}"
