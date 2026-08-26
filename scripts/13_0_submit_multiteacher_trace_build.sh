#!/bin/bash
# Submit MATH trajectory generation for the missing teachers and the audited mixed-data build.

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/slurm/_gpu_idle_gate.sh

CONFIG="${CONFIG:-configs/capacity_length_multibench_multiteacher_kd_pilot_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_multibench_multiteacher_kd_pilot_v1}"
BEEGFS_RESULT_ROOT="${BEEGFS_RESULT_ROOT:-/mnt/beegfs/${USER}/projects/small_language_model/results/capacity_length_multibench_multiteacher_kd_pilot_v1}"
UPSTREAM_KD_AUDIT_JOB="${UPSTREAM_KD_AUDIT_JOB:-}"
GENERATION_SHARDS_1P5B="${GENERATION_SHARDS_1P5B:-1}"
GENERATION_SHARDS_3B="${GENERATION_SHARDS_3B:-2}"
GENERATION_SHARDS_14B="${GENERATION_SHARDS_14B:-2}"
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

small_dependency=()
if [ -n "${UPSTREAM_KD_AUDIT_JOB}" ]; then
  small_dependency=("--dependency=afterok:${UPSTREAM_KD_AUDIT_JOB}")
fi

job_1p5b=$(submit DRY_GEN_1P5B \
  sbatch --parsable -p a5000ada -w c32 "${small_dependency[@]}" \
  --export="ALL,CONFIG=${CONFIG},OUTPUT_ROOT=${OUTPUT_ROOT},GENERATOR_NAME=qwen2p5_1p5b,GENERATION_SHARDS=${GENERATION_SHARDS_1P5B},MIN_FREE_MIB=12000,GENERATION_ENV=/mnt/beegfs/youyang7/.conda/envs/fact" \
  scripts/slurm/13_1_generate_multiteacher_math.sh | tail -n 1)

job_3b=$(submit DRY_GEN_3B \
  sbatch --parsable -p a6000 -w c31 \
  --export="ALL,CONFIG=${CONFIG},OUTPUT_ROOT=${OUTPUT_ROOT},GENERATOR_NAME=qwen2p5_3b,GENERATION_SHARDS=${GENERATION_SHARDS_3B},MIN_FREE_MIB=30000,GENERATION_ENV=/mnt/beegfs/youyang7/.conda/envs/fact" \
  scripts/slurm/13_1_generate_multiteacher_math.sh | tail -n 1)

job_14b=$(submit DRY_GEN_14B \
  sbatch --parsable -p a6000 -w c31 --dependency="afterok:${job_3b}" \
  --export="ALL,CONFIG=${CONFIG},OUTPUT_ROOT=${OUTPUT_ROOT},GENERATOR_NAME=qwen2p5_14b,GENERATION_SHARDS=${GENERATION_SHARDS_14B},MIN_FREE_MIB=45000,GENERATION_ENV=/mnt/beegfs/youyang7/.conda/envs/fact" \
  scripts/slurm/13_1_generate_multiteacher_math.sh | tail -n 1)

generation_dependency="afterok:${job_1p5b}:${job_3b}:${job_14b}"
build_job=$(submit DRY_BUILD \
  sbatch --parsable -p a6000 -w c31 --dependency="${generation_dependency}" \
  --export="ALL,CONFIG=${CONFIG},OUTPUT_ROOT=${OUTPUT_ROOT},BUILD_ENV=/mnt/beegfs/youyang7/.conda/envs/fact" \
  scripts/slurm/13_2_select_build_multiteacher_multibench_kd.sh | tail -n 1)

echo "generation_jobs=${job_1p5b},${job_3b},${job_14b} build_job=${build_job}"
