#!/bin/bash
# Launch the ranked-length generation pipeline inside an existing C49 allocation.

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/slurm/_gpu_idle_gate.sh

ALLOCATION_JOB_ID="${ALLOCATION_JOB_ID:-276695}"
CONFIG="${CONFIG:-configs/capacity_length_ranked_sampling_7b_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_ranked_sampling_7b_v1}"
BEEGFS_RESULT_ROOT="${BEEGFS_RESULT_ROOT:-/mnt/beegfs/${USER}/projects/small_language_model/results/capacity_length_ranked_sampling_7b_v1}"
MIN_FREE_MIB="${MIN_FREE_MIB:-25000}"
RUN_TAG="${RUN_TAG:-c49_${ALLOCATION_JOB_ID}_$(date -u +%Y%m%dT%H%M%SZ)}"

if [ ! -f "${CONFIG}" ]; then
  echo "Missing ranked-sampling config: ${CONFIG}" >&2
  exit 2
fi
job_state=$(squeue -h -j "${ALLOCATION_JOB_ID}" -o '%T')
job_node=$(squeue -h -j "${ALLOCATION_JOB_ID}" -o '%N')
job_user=$(squeue -h -j "${ALLOCATION_JOB_ID}" -o '%u')
if [ "${job_state}" != "RUNNING" ] || [ "${job_node}" != "c49" ] || [ "${job_user}" != "${USER}" ]; then
  echo "Job ${ALLOCATION_JOB_ID} is not the user's active C49 allocation." >&2
  exit 2
fi

if [ -L "${OUTPUT_ROOT}" ]; then
  resolved=$(readlink -f "${OUTPUT_ROOT}")
  if [ "${resolved}" != "${BEEGFS_RESULT_ROOT}" ]; then
    echo "Existing result symlink targets ${resolved}, expected ${BEEGFS_RESULT_ROOT}." >&2
    exit 2
  fi
elif [ -e "${OUTPUT_ROOT}" ]; then
  echo "Refusing to replace existing non-symlink result path: ${OUTPUT_ROOT}" >&2
  exit 2
else
  mkdir_with_retry "${BEEGFS_RESULT_ROOT}"
  ln -s "${BEEGFS_RESULT_ROOT}" "${OUTPUT_ROOT}"
fi
if [ -e "${OUTPUT_ROOT}/formal/datasets/GENERATION_COMPLETE" ]; then
  echo "Generation is already complete: ${OUTPUT_ROOT}/formal/datasets/GENERATION_COMPLETE" >&2
  exit 2
fi
mkdir_with_retry logs

runner_log="logs/16_ranked_length_${RUN_TAG}.out"
runner_pid_path="logs/16_ranked_length_${RUN_TAG}.pid"
nohup srun --jobid="${ALLOCATION_JOB_ID}" --overlap -N1 -n1 --cpus-per-task=16 \
  --job-name=16_ranked_length_c49 \
  --export="ALL,CONFIG=${CONFIG},OUTPUT_ROOT=${OUTPUT_ROOT},MIN_FREE_MIB=${MIN_FREE_MIB},RUN_TAG=${RUN_TAG}" \
  bash scripts/slurm/16_0_run_ranked_length_c49.sh \
  >"${runner_log}" 2>&1 < /dev/null &
runner_pid="$!"
printf '%s\n' "${runner_pid}" >"${runner_pid_path}"
echo "allocation_job=${ALLOCATION_JOB_ID} runner_pid=${runner_pid} run_tag=${RUN_TAG} log=${runner_log}"
