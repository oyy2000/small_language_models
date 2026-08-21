#!/bin/bash
# Submit four training shards, four dependent evaluation shards, and analysis.

set -euo pipefail
cd "$(dirname "$0")/.."

STAGE="${STAGE:-formal}"
CONFIG="${CONFIG:-configs/capacity_length_factorial_v1.json}"
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/capacity_length_factorial_sft_v1.json}"
EVALUATION_CONFIG="${EVALUATION_CONFIG:-configs/capacity_length_factorial_eval_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_factorial_v1}"
UPSTREAM_JOB_ID="${UPSTREAM_JOB_ID:-}"
DRY_RUN="${DRY_RUN:-0}"
SERIAL_NODE="${SERIAL_NODE:-}"
SERIAL_PARTITION="${SERIAL_PARTITION:-}"
if [ -n "${SERIAL_NODE}" ]; then
  if [ -z "${SERIAL_PARTITION}" ]; then
    echo "SERIAL_PARTITION is required when SERIAL_NODE is set." >&2
    exit 2
  fi
  nodes=("${SERIAL_NODE}" "${SERIAL_NODE}" "${SERIAL_NODE}" "${SERIAL_NODE}")
  partitions=("${SERIAL_PARTITION}" "${SERIAL_PARTITION}" "${SERIAL_PARTITION}" "${SERIAL_PARTITION}")
else
  nodes=(c30 c31 c32 c49)
  partitions=(a6000 a6000 a5000ada a5000ada)
fi

train_jobs=()
previous_job="${UPSTREAM_JOB_ID}"
for index in "${!nodes[@]}"; do
  dependency_args=()
  if [ -n "${previous_job}" ]; then
    dependency_args=(--dependency="afterok:${previous_job}")
  fi
  command=(
    sbatch --parsable -p "${partitions[$index]}" -w "${nodes[$index]}"
    "${dependency_args[@]}"
    --export="ALL,STAGE=${STAGE},CONFIG=${CONFIG},TRAINING_CONFIG=${TRAINING_CONFIG},OUTPUT_ROOT=${OUTPUT_ROOT},LAUNCHER_SHARDS=4,LAUNCHER_SHARD_INDEX=${index}"
    scripts/slurm/6_1_train_capacity_length_students.sh
  )
  if [ "${DRY_RUN}" = "1" ]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    train_jobs+=("DRY_TRAIN_${index}")
  else
    train_jobs+=("$("${command[@]}")")
  fi
  if [ -n "${SERIAL_NODE}" ]; then
    previous_job="${train_jobs[-1]}"
  else
    previous_job="${UPSTREAM_JOB_ID}"
  fi
done

if [ -n "${SERIAL_NODE}" ]; then
  train_dependency="afterok:${train_jobs[-1]}"
else
  train_dependency="afterok:$(IFS=:; echo "${train_jobs[*]}")"
fi
eval_jobs=()
previous_eval_job=""
for index in "${!nodes[@]}"; do
  if [ -n "${SERIAL_NODE}" ] && [ -n "${previous_eval_job}" ]; then
    eval_dependency="afterok:${previous_eval_job}"
  else
    eval_dependency="${train_dependency}"
  fi
  command=(
    sbatch --parsable -p "${partitions[$index]}" -w "${nodes[$index]}"
    --dependency="${eval_dependency}"
    --export="ALL,STAGE=${STAGE},CONFIG=${CONFIG},EVALUATION_CONFIG=${EVALUATION_CONFIG},OUTPUT_ROOT=${OUTPUT_ROOT},LAUNCHER_SHARDS=4,LAUNCHER_SHARD_INDEX=${index}"
    scripts/slurm/7_1_eval_capacity_length_students.sh
  )
  if [ "${DRY_RUN}" = "1" ]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    eval_jobs+=("DRY_EVAL_${index}")
  else
    eval_jobs+=("$("${command[@]}")")
  fi
  if [ -n "${SERIAL_NODE}" ]; then
    previous_eval_job="${eval_jobs[-1]}"
  fi
done

if [ -n "${SERIAL_NODE}" ]; then
  eval_dependency="afterok:${eval_jobs[-1]}"
else
  eval_dependency="afterok:$(IFS=:; echo "${eval_jobs[*]}")"
fi
analysis_command=(
  sbatch --parsable --dependency="${eval_dependency}"
  --export="ALL,STAGE=${STAGE},CONFIG=${CONFIG},OUTPUT_ROOT=${OUTPUT_ROOT}"
  scripts/slurm/7_2_analyze_capacity_length_interaction.sh
)
if [ "${DRY_RUN}" = "1" ]; then
  printf '%q ' "${analysis_command[@]}"
  printf '\n'
  analysis_job="DRY_ANALYSIS"
else
  analysis_job=$("${analysis_command[@]}")
fi

audit_command=(
  sbatch --parsable --dependency="afterok:${analysis_job}"
  --export="ALL,STAGE=${STAGE},CONFIG=${CONFIG},OUTPUT_ROOT=${OUTPUT_ROOT}"
  scripts/slurm/8_1_audit_capacity_length_completion.sh
)
if [ "${DRY_RUN}" = "1" ]; then
  printf '%q ' "${audit_command[@]}"
  printf '\n'
else
  audit_job=$("${audit_command[@]}")
  echo "training_jobs=$(IFS=,; echo "${train_jobs[*]}") evaluation_jobs=$(IFS=,; echo "${eval_jobs[*]}") analysis_job=${analysis_job} audit_job=${audit_job}"
fi
