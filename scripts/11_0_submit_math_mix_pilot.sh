#!/bin/bash
# Submit the complete exploratory MATH-mix pilot DAG without modifying frozen formal artifacts.

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/slurm/_gpu_idle_gate.sh

CONFIG="${CONFIG:-configs/capacity_length_math_mix_pilot_v1.json}"
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/capacity_length_math_mix_pilot_sft_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_math_mix_pilot_v1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/capacity_length_math_mix_pilot_v1/pilot}"
FIGURE_ROOT="${FIGURE_ROOT:-figures/capacity_length_math_mix_pilot_v1}"
BEEGFS_RESULT_ROOT="${BEEGFS_RESULT_ROOT:-/mnt/beegfs/${USER}/projects/small_language_model/results/capacity_length_math_mix_pilot_v1}"
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
  mkdir_with_retry "${CHECKPOINT_ROOT}"
  mkdir -p "${FIGURE_ROOT}" logs
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

prep_job=$(submit DRY_PREP \
  sbatch --parsable -p a6000 -w c31 \
  --export="ALL,CONFIG=${CONFIG},OUTPUT_ROOT=${OUTPUT_ROOT},PREP_ENV=/mnt/beegfs/youyang7/.conda/envs/fact" \
  scripts/slurm/11_0_prepare_math_mix_pilot.sh | tail -n 1)

generation_job=$(submit DRY_GENERATION \
  sbatch --parsable -p a6000 -w c31 --dependency="afterok:${prep_job}" \
  --export="ALL,CONFIG=${CONFIG},OUTPUT_ROOT=${OUTPUT_ROOT},GENERATION_ENV=/mnt/beegfs/youyang7/.conda/envs/fact" \
  scripts/slurm/11_1_generate_math_mix_pilot.sh | tail -n 1)

build_job=$(submit DRY_BUILD \
  sbatch --parsable -p a6000 -w c31 --dependency="afterok:${generation_job}" \
  --export="ALL,CONFIG=${CONFIG},OUTPUT_ROOT=${OUTPUT_ROOT},BUILD_ENV=/mnt/beegfs/youyang7/.conda/envs/fact" \
  scripts/slurm/11_2_merge_build_math_mix_pilot.sh | tail -n 1)

# Keep one launcher per node so each idle-GPU gate has a disjoint physical
# inventory.  c30/c49 are intentionally excluded from this pilot because they
# are commonly occupied by long-running allocations; each launcher can process
# multiple assigned runs sequentially as its selected GPUs become available.
nodes=(c31 c32)
partitions=(a6000 a5000ada)
launcher_shards="${#nodes[@]}"
train_jobs=()
for index in "${!nodes[@]}"; do
  train_jobs+=("$(submit "DRY_TRAIN_${index}" \
    sbatch --parsable -p "${partitions[$index]}" -w "${nodes[$index]}" \
    --dependency="afterok:${build_job}" \
    --export="ALL,STAGE=pilot,CONFIG=${CONFIG},TRAINING_CONFIG=${TRAINING_CONFIG},OUTPUT_ROOT=${OUTPUT_ROOT},CHECKPOINT_ROOT=${CHECKPOINT_ROOT},LAUNCHER_SHARDS=${launcher_shards},LAUNCHER_SHARD_INDEX=${index},SFT_ENV=/mnt/beegfs/youyang7/.conda/envs/sft" \
    scripts/slurm/6_1_train_capacity_length_students.sh | tail -n 1)")
done

train_dependency="afterok:$(IFS=:; echo "${train_jobs[*]}")"
eval_jobs=()
for index in "${!nodes[@]}"; do
  eval_jobs+=("$(submit "DRY_EVAL_${index}" \
    sbatch --parsable -p "${partitions[$index]}" -w "${nodes[$index]}" \
    --dependency="${train_dependency}" \
    --export="ALL,CONFIG=${CONFIG},OUTPUT_ROOT=${OUTPUT_ROOT},LAUNCHER_SHARDS=${launcher_shards},LAUNCHER_SHARD_INDEX=${index},EVAL_ENV=/mnt/beegfs/youyang7/.conda/envs/fact" \
    scripts/slurm/11_5_eval_math_mix_pilot.sh | tail -n 1)")
done

eval_dependency="afterok:$(IFS=:; echo "${eval_jobs[*]}")"
analysis_job=$(submit DRY_ANALYSIS \
  sbatch --parsable --dependency="${eval_dependency}" \
  --export="ALL,OUTPUT_ROOT=${OUTPUT_ROOT},FIGURE_ROOT=${FIGURE_ROOT},ANALYSIS_ENV=/mnt/beegfs/youyang7/.conda/envs/fact" \
  scripts/slurm/11_6_analyze_math_mix_pilot.sh | tail -n 1)

audit_job=$(submit DRY_AUDIT \
  sbatch --parsable --dependency="afterok:${analysis_job}" \
  --export="ALL,OUTPUT_ROOT=${OUTPUT_ROOT},FIGURE_ROOT=${FIGURE_ROOT},AUDIT_ENV=/mnt/beegfs/youyang7/.conda/envs/fact" \
  scripts/slurm/11_7_audit_math_mix_pilot.sh | tail -n 1)

echo "prep_job=${prep_job} generation_job=${generation_job} build_job=${build_job}"
echo "training_jobs=$(IFS=,; echo "${train_jobs[*]}")"
echo "evaluation_jobs=$(IFS=,; echo "${eval_jobs[*]}") analysis_job=${analysis_job} audit_job=${audit_job}"
