#!/bin/bash
# Submit multi-GPU ranked-length generation and a dependent audited merge.

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/slurm/_gpu_idle_gate.sh

CONFIG="${CONFIG:-configs/capacity_length_ranked_sampling_7b_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_ranked_sampling_7b_v1}"
BEEGFS_RESULT_ROOT="${BEEGFS_RESULT_ROOT:-/mnt/beegfs/${USER}/projects/small_language_model/results/capacity_length_ranked_sampling_7b_v1}"
GENERATION_NODE="${GENERATION_NODE:-c31}"
GENERATION_PARTITION="${GENERATION_PARTITION:-a6000}"
REQUIRED_GPUS="${REQUIRED_GPUS:-3}"
MIN_FREE_MIB="${MIN_FREE_MIB:-45000}"
DRY_RUN="${DRY_RUN:-0}"

if [ ! -f "${CONFIG}" ]; then
  echo "Missing ranked-sampling config: ${CONFIG}" >&2
  exit 2
fi
config_shards=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["generation"]["num_shards"])' "${CONFIG}")
if [ "${config_shards}" -ne "${REQUIRED_GPUS}" ]; then
  echo "Config num_shards=${config_shards} does not match REQUIRED_GPUS=${REQUIRED_GPUS}." >&2
  exit 2
fi

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
  if [ -e "${OUTPUT_ROOT}/formal/datasets/GENERATION_COMPLETE" ]; then
    echo "Generation is already complete: ${OUTPUT_ROOT}/formal/datasets/GENERATION_COMPLETE" >&2
    exit 2
  fi
  mkdir_with_retry logs
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

smoke_job=$(submit DRY_RANKED_SMOKE \
  sbatch --parsable -p "${GENERATION_PARTITION}" -w "${GENERATION_NODE}" \
  --export="ALL,CONFIG=${CONFIG},OUTPUT_ROOT=${OUTPUT_ROOT},MIN_FREE_MIB=${MIN_FREE_MIB},GENERATION_ENV=/mnt/beegfs/youyang7/.conda/envs/fact" \
  scripts/slurm/16_0_smoke_ranked_length_sample.sh | tail -n 1)

generation_job=$(submit DRY_RANKED_GEN \
  sbatch --parsable -p "${GENERATION_PARTITION}" -w "${GENERATION_NODE}" \
  --dependency="afterok:${smoke_job}" \
  --export="ALL,CONFIG=${CONFIG},OUTPUT_ROOT=${OUTPUT_ROOT},REQUIRED_GPUS=${REQUIRED_GPUS},MIN_FREE_MIB=${MIN_FREE_MIB},GENERATION_ENV=/mnt/beegfs/youyang7/.conda/envs/fact" \
  scripts/slurm/16_1_generate_ranked_length_samples.sh | tail -n 1)

merge_job=$(submit DRY_RANKED_MERGE \
  sbatch --parsable -p "${GENERATION_PARTITION}" -w "${GENERATION_NODE}" \
  --dependency="afterok:${generation_job}" \
  --export="ALL,CONFIG=${CONFIG},OUTPUT_ROOT=${OUTPUT_ROOT},SFT_ENV=/mnt/beegfs/youyang7/.conda/envs/sft" \
  scripts/slurm/16_2_merge_ranked_length_samples.sh | tail -n 1)

echo "ranked_smoke_job=${smoke_job} ranked_generation_job=${generation_job} ranked_merge_job=${merge_job} node=${GENERATION_NODE} shards=${REQUIRED_GPUS}"
