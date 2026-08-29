#!/bin/bash
# Submit the seed-42/73 ranked-length extension with a physical-GPU admission gate.

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/slurm/_gpu_idle_gate.sh

TRAINING_CONFIG="${TRAINING_CONFIG:-configs/capacity_length_ranked_sampling_7b_training_seed42_73_v1.json}"
TRAINING_OVERLAY="${TRAINING_OVERLAY:-configs/capacity_length_ranked_sampling_7b_sft_seed42_73_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_ranked_sampling_7b_multiseed_v1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/capacity_length_ranked_sampling_7b_multiseed_v1}"
BEEGFS_CHECKPOINT_ROOT="${BEEGFS_CHECKPOINT_ROOT:-/mnt/beegfs/${USER}/projects/small_language_model/checkpoints/capacity_length_ranked_sampling_7b_multiseed_v1}"
PARTITION="${PARTITION:-a5000ada}"
NODE="${NODE:-c32}"
MIN_FREE_MIB="${MIN_FREE_MIB:-18000}"
MIN_GPUS="${MIN_GPUS:-3}"
DRY_RUN="${DRY_RUN:-0}"

for path in "${TRAINING_CONFIG}" "${TRAINING_OVERLAY}" \
  results/capacity_length_ranked_sampling_7b_v1/formal/datasets/GENERATION_COMPLETE; do
  if [ ! -f "${path}" ]; then
    echo "Required input is missing: ${path}" >&2
    exit 2
  fi
done
if [ -e "${OUTPUT_ROOT}/formal/training/seed42_73/audit/TRAINING_COMPLETE" ]; then
  echo "Seed-42/73 training is already complete: ${OUTPUT_ROOT}" >&2
  exit 2
fi

if [ "${DRY_RUN}" != "1" ]; then
  mkdir_with_retry "${BEEGFS_CHECKPOINT_ROOT}"
  expected_resolved=$(readlink -f "${BEEGFS_CHECKPOINT_ROOT}")
  if [ -L "${CHECKPOINT_ROOT}" ]; then
    resolved=$(readlink -f "${CHECKPOINT_ROOT}")
    if [ "${resolved}" != "${expected_resolved}" ]; then
      echo "Checkpoint symlink targets ${resolved}, expected ${expected_resolved}." >&2
      exit 2
    fi
  elif [ -d "${CHECKPOINT_ROOT}" ]; then
    resolved=$(readlink -f "${CHECKPOINT_ROOT}")
    if [ "${resolved}" != "${expected_resolved}" ]; then
      echo "Checkpoint directory resolves to ${resolved}, expected ${expected_resolved}." >&2
      exit 2
    fi
  elif [ -e "${CHECKPOINT_ROOT}" ]; then
    echo "Refusing non-directory checkpoint path: ${CHECKPOINT_ROOT}" >&2
    exit 2
  else
    ln -s "${BEEGFS_CHECKPOINT_ROOT}" "${CHECKPOINT_ROOT}"
  fi
  mkdir_with_retry logs
fi

command=(
  sbatch --parsable -p "${PARTITION}" -w "${NODE}"
  --export="ALL,TRAINING_CONFIG=${TRAINING_CONFIG},TRAINING_OVERLAY=${TRAINING_OVERLAY},OUTPUT_ROOT=${OUTPUT_ROOT},CHECKPOINT_ROOT=${CHECKPOINT_ROOT},MIN_FREE_MIB=${MIN_FREE_MIB},MIN_GPUS=${MIN_GPUS},SFT_ENV=/mnt/beegfs/youyang7/.conda/envs/sft"
  scripts/slurm/18_2_train_ranked_multiseed.sh
)
if [ "${DRY_RUN}" = "1" ]; then
  printf '%q ' "${command[@]}"
  printf '\n'
else
  job_id=$("${command[@]}" | tail -n 1)
  echo "training_job=${job_id} partition=${PARTITION} node=${NODE} output_root=${OUTPUT_ROOT}"
fi
