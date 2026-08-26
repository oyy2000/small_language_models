#!/bin/bash
# Submit matched SFT/logit-KD training and frozen multi-benchmark evaluation.

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/capacity_length_multibench_multiteacher_kd_pilot_v1.json}"
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/capacity_length_multibench_multiteacher_kd_pilot_sft_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_multibench_multiteacher_kd_pilot_v1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/capacity_length_multibench_multiteacher_kd_pilot_v1/pilot}"
BUILD_JOB_ID="${BUILD_JOB_ID:?BUILD_JOB_ID is required}"
DRY_RUN="${DRY_RUN:-0}"

if [ "${DRY_RUN}" != "1" ]; then
  mkdir -p "${CHECKPOINT_ROOT}" logs
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

freeze_job=$(submit DRY_FREEZE \
  sbatch --parsable -p a6000 -w c31 --dependency="afterok:${BUILD_JOB_ID}" \
  --export="ALL,CONFIG=${CONFIG},OUTPUT_ROOT=${OUTPUT_ROOT},PREP_ENV=/mnt/beegfs/youyang7/.conda/envs/fact" \
  scripts/slurm/13_3_freeze_multiteacher_kd_protocol.sh | tail -n 1)

nodes=(c31 c32)
partitions=(a6000 a5000ada)
launcher_shards="${#nodes[@]}"
sft_jobs=()
for index in "${!nodes[@]}"; do
  sft_jobs+=("$(submit "DRY_SFT_${index}" \
    sbatch --parsable -p "${partitions[$index]}" -w "${nodes[$index]}" \
    --dependency="afterok:${BUILD_JOB_ID}" \
    --export="ALL,STAGE=pilot,CONFIG=${CONFIG},TRAINING_CONFIG=${TRAINING_CONFIG},OUTPUT_ROOT=${OUTPUT_ROOT},CHECKPOINT_ROOT=${CHECKPOINT_ROOT}/sft,LAUNCHER_SHARDS=${launcher_shards},LAUNCHER_SHARD_INDEX=${index},SFT_ENV=/mnt/beegfs/youyang7/.conda/envs/sft" \
    scripts/slurm/6_1_train_capacity_length_students.sh | tail -n 1)")
done
sft_dependency="afterok:$(IFS=:; echo "${sft_jobs[*]}")"

kd_14b_job=$(submit DRY_KD_14B \
  sbatch --parsable -p a6000 -w c30 --dependency="afterok:${freeze_job}" \
  --export="ALL,OUTPUT_ROOT=${OUTPUT_ROOT},TEACHER_NAMES=qwen2p5_14b,MIN_FREE_MIB=40000,SFT_ENV=/mnt/beegfs/youyang7/.conda/envs/sft" \
  scripts/slurm/13_5_train_multiteacher_logit_kd.sh | tail -n 1)

small_kd_dependency="afterok:${freeze_job}:$(IFS=:; echo "${sft_jobs[*]}")"
kd_small_job=$(submit DRY_KD_SMALL \
  sbatch --parsable -p a5000ada -w c32 --dependency="${small_kd_dependency}" \
  --export="ALL,OUTPUT_ROOT=${OUTPUT_ROOT},TEACHER_NAMES=qwen2p5_1p5b:qwen2p5_3b,MIN_FREE_MIB=24000,SFT_ENV=/mnt/beegfs/youyang7/.conda/envs/sft" \
  scripts/slurm/13_5_train_multiteacher_logit_kd.sh | tail -n 1)

kd_7b_job=$(submit DRY_KD_7B \
  sbatch --parsable -p a6000 -w c31 --dependency="${small_kd_dependency}" \
  --export="ALL,OUTPUT_ROOT=${OUTPUT_ROOT},TEACHER_NAMES=qwen2p5_7b,MIN_FREE_MIB=28000,SFT_ENV=/mnt/beegfs/youyang7/.conda/envs/sft" \
  scripts/slurm/13_5_train_multiteacher_logit_kd.sh | tail -n 1)

all_training_dependency="afterok:$(IFS=:; echo "${sft_jobs[*]}"):${kd_14b_job}:${kd_small_job}:${kd_7b_job}"
registry_job=$(submit DRY_REGISTRY \
  sbatch --parsable -p a6000 -w c31 --dependency="${all_training_dependency}" \
  --export="ALL,OUTPUT_ROOT=${OUTPUT_ROOT},PREP_ENV=/mnt/beegfs/youyang7/.conda/envs/fact" \
  scripts/slurm/13_6_prepare_multiteacher_model_registry.sh | tail -n 1)

eval_jobs=()
for index in "${!nodes[@]}"; do
  eval_jobs+=("$(submit "DRY_EVAL_${index}" \
    sbatch --parsable -p "${partitions[$index]}" -w "${nodes[$index]}" \
    --dependency="afterok:${registry_job}" \
    --export="ALL,CONFIG=${CONFIG},OUTPUT_ROOT=${OUTPUT_ROOT},LAUNCHER_SHARDS=${launcher_shards},LAUNCHER_SHARD_INDEX=${index},EVAL_ENV=/mnt/beegfs/youyang7/.conda/envs/fact" \
    scripts/slurm/13_7_eval_multiteacher_multibench_kd.sh | tail -n 1)")
done

echo "freeze_job=${freeze_job} sft_jobs=$(IFS=,; echo "${sft_jobs[*]}") kd_jobs=${kd_14b_job},${kd_small_job},${kd_7b_job} registry_job=${registry_job} eval_jobs=$(IFS=,; echo "${eval_jobs[*]}")"
