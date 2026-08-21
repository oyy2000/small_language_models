#!/bin/bash
# Submit four generator jobs and one dependent merge/data-build job.

set -euo pipefail
cd "$(dirname "$0")/.."

STAGE="${STAGE:-smoke}"
CONFIG="${CONFIG:-configs/capacity_length_factorial_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_factorial_v1}"
DRY_RUN="${DRY_RUN:-0}"
source scripts/slurm/_gpu_idle_gate.sh

if [ "${DRY_RUN}" != "1" ]; then
  for generator in qwen2p5_14b qwen2p5_7b qwen2p5_3b qwen2p5_1p5b; do
    mkdir_with_retry "${OUTPUT_ROOT}/${STAGE}/raw/${generator}"
  done
fi

submit_generator() {
  local partition="$1"
  local node="$2"
  local generator="$3"
  local min_free_mib="$4"
  local command=(
    sbatch --parsable -p "${partition}" -w "${node}"
    --export="ALL,STAGE=${STAGE},CONFIG=${CONFIG},OUTPUT_ROOT=${OUTPUT_ROOT},GENERATOR_NAME=${generator},MIN_FREE_MIB=${min_free_mib}"
    scripts/slurm/5_1_generate_capacity_length_factorial.sh
  )
  if [ "${DRY_RUN}" = "1" ]; then
    printf '%q ' "${command[@]}" >&2
    printf '\n' >&2
    echo "DRY_${generator}"
  else
    "${command[@]}"
  fi
}

job_14b=$(submit_generator a6000 c30 qwen2p5_14b 40000 | tail -n 1)
job_7b=$(submit_generator a6000 c31 qwen2p5_7b 20000 | tail -n 1)
job_3b=$(submit_generator a5000ada c32 qwen2p5_3b 12000 | tail -n 1)
job_1p5b=$(submit_generator a5000ada c49 qwen2p5_1p5b 10000 | tail -n 1)

dependency="afterok:${job_14b}:${job_7b}:${job_3b}:${job_1p5b}"
merge_command=(
  sbatch --parsable --dependency="${dependency}"
  --export="ALL,STAGE=${STAGE},CONFIG=${CONFIG},OUTPUT_ROOT=${OUTPUT_ROOT}"
  scripts/slurm/5_2_merge_select_build_capacity_length.sh
)
if [ "${DRY_RUN}" = "1" ]; then
  printf '%q ' "${merge_command[@]}"
  printf '\n'
else
  merge_job=$("${merge_command[@]}")
  echo "generation_jobs=${job_14b},${job_7b},${job_3b},${job_1p5b} merge_job=${merge_job} stage=${STAGE}"
fi
