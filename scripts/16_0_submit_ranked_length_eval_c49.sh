#!/bin/bash
# Launch ranked-length evaluation and reporting in the existing C49 allocation.

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/slurm/_gpu_idle_gate.sh

ALLOCATION_JOB_ID="${ALLOCATION_JOB_ID:-276695}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_ranked_sampling_7b_v1}"
FIGURE_ROOT="${FIGURE_ROOT:-figures/capacity_length_ranked_sampling_7b_v1}"
MIN_FREE_MIB="${MIN_FREE_MIB:-16000}"
RUN_TAG="${RUN_TAG:-c49_eval_${ALLOCATION_JOB_ID}_$(date -u +%Y%m%dT%H%M%SZ)}"

job_state=$(squeue -h -j "${ALLOCATION_JOB_ID}" -o '%T')
job_node=$(squeue -h -j "${ALLOCATION_JOB_ID}" -o '%N')
job_user=$(squeue -h -j "${ALLOCATION_JOB_ID}" -o '%u')
if [ "${job_state}" != "RUNNING" ] || [ "${job_node}" != "c49" ] || [ "${job_user}" != "${USER}" ]; then
  echo "Job ${ALLOCATION_JOB_ID} is not the user's active C49 allocation." >&2
  exit 2
fi
if [ ! -f "${OUTPUT_ROOT}/formal/training/audit/TRAINING_COMPLETE" ]; then
  echo "Ranked-length training is not complete: ${OUTPUT_ROOT}" >&2
  exit 2
fi
if [ -e "${OUTPUT_ROOT}/FORMAL_COMPLETE" ]; then
  echo "Ranked-length experiment is already sealed: ${OUTPUT_ROOT}/FORMAL_COMPLETE" >&2
  exit 2
fi
mkdir_with_retry logs
mkdir_with_retry "${FIGURE_ROOT}/formal"

runner_log="logs/16_ranked_length_${RUN_TAG}.out"
runner_pid_path="logs/16_ranked_length_${RUN_TAG}.pid"
nohup srun --jobid="${ALLOCATION_JOB_ID}" --overlap -N1 -n1 --cpus-per-task=16 \
  --job-name=16_ranked_eval \
  --export="ALL,OUTPUT_ROOT=${OUTPUT_ROOT},FIGURE_ROOT=${FIGURE_ROOT},MIN_FREE_MIB=${MIN_FREE_MIB},RUN_TAG=${RUN_TAG}" \
  bash scripts/slurm/16_6_eval_analyze_ranked_length_c49.sh \
  >"${runner_log}" 2>&1 < /dev/null &
runner_pid="$!"
printf '%s\n' "${runner_pid}" >"${runner_pid_path}"
echo "allocation_job=${ALLOCATION_JOB_ID} runner_pid=${runner_pid} run_tag=${RUN_TAG} log=${runner_log}"
