#!/bin/bash
# Submit independent phase-14 paired rewrite generation and audited data build.

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/slurm/_gpu_idle_gate.sh

CONFIG="${CONFIG:-configs/capacity_length_paired_rewrite_7b_pilot_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_paired_rewrite_7b_pilot_v1}"
BEEGFS_RESULT_ROOT="${BEEGFS_RESULT_ROOT:-/mnt/beegfs/${USER}/projects/small_language_model/results/capacity_length_paired_rewrite_7b_pilot_v1}"
DRY_RUN="${DRY_RUN:-0}"

if [ "${DRY_RUN}" != "1" ]; then
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
  mkdir -p logs
fi

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

generation_job=$(submit DRY_REWRITE_GEN \
  sbatch --parsable -p a6000 -w c31 \
  --export="ALL,CONFIG=${CONFIG},OUTPUT_ROOT=${OUTPUT_ROOT},MIN_FREE_MIB=26000,GENERATION_ENV=/mnt/beegfs/youyang7/.conda/envs/fact" \
  scripts/slurm/14_1_generate_paired_rewrites.sh | tail -n 1)
build_job=$(submit DRY_REWRITE_BUILD \
  sbatch --parsable -p a6000 -w c31 --dependency="afterok:${generation_job}" \
  --export="ALL,CONFIG=${CONFIG},OUTPUT_ROOT=${OUTPUT_ROOT},BUILD_ENV=/mnt/beegfs/youyang7/.conda/envs/sft" \
  scripts/slurm/14_2_build_paired_rewrite_sft_data.sh | tail -n 1)
echo "paired_rewrite_generation_job=${generation_job} paired_rewrite_build_job=${build_job}"
