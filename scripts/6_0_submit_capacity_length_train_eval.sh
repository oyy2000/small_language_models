#!/bin/bash
# Submit four training shards, four dependent evaluation shards, and analysis.

set -euo pipefail
cd "$(dirname "$0")/.."

STAGE="${STAGE:-formal}"
CONFIG="${CONFIG:-configs/capacity_length_factorial_v1.json}"
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/capacity_length_factorial_sft_v1.json}"
EVALUATION_CONFIG="${EVALUATION_CONFIG:-configs/capacity_length_factorial_eval_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_factorial_v1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/capacity_length_factorial_v1/${STAGE}}"
UPSTREAM_JOB_ID="${UPSTREAM_JOB_ID:-}"
DRY_RUN="${DRY_RUN:-0}"
SERIAL_NODE="${SERIAL_NODE:-}"
SERIAL_PARTITION="${SERIAL_PARTITION:-}"
NODE_CSV="${NODE_CSV:-}"
PARTITION_CSV="${PARTITION_CSV:-}"
if [ -n "${NODE_CSV}" ]; then
  if [ -z "${PARTITION_CSV}" ]; then
    echo "PARTITION_CSV is required when NODE_CSV is set." >&2
    exit 2
  fi
  IFS=, read -r -a nodes <<< "${NODE_CSV}"
  IFS=, read -r -a partitions <<< "${PARTITION_CSV}"
elif [ -n "${SERIAL_NODE}" ]; then
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
if [ "${#nodes[@]}" -ne 4 ] || [ "${#partitions[@]}" -ne 4 ]; then
  echo "Exactly four node and partition entries are required." >&2
  exit 2
fi

train_jobs=()
declare -A last_train_by_node=()
for index in "${!nodes[@]}"; do
  dependency_args=()
  predecessor="${last_train_by_node[${nodes[$index]}]:-${UPSTREAM_JOB_ID}}"
  if [ -n "${predecessor}" ]; then
    dependency_args=(--dependency="afterok:${predecessor}")
  fi
  command=(
    sbatch --parsable -p "${partitions[$index]}" -w "${nodes[$index]}"
    "${dependency_args[@]}"
    --export="ALL,STAGE=${STAGE},CONFIG=${CONFIG},TRAINING_CONFIG=${TRAINING_CONFIG},OUTPUT_ROOT=${OUTPUT_ROOT},CHECKPOINT_ROOT=${CHECKPOINT_ROOT},LAUNCHER_SHARDS=4,LAUNCHER_SHARD_INDEX=${index}"
    scripts/slurm/6_1_train_capacity_length_students.sh
  )
  if [ "${DRY_RUN}" = "1" ]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    train_jobs+=("DRY_TRAIN_${index}")
  else
    train_jobs+=("$("${command[@]}")")
  fi
  last_train_by_node[${nodes[$index]}]="${train_jobs[-1]}"
done

train_leaf_jobs=()
declare -A seen_train_nodes=()
for node in "${nodes[@]}"; do
  if [ -z "${seen_train_nodes[${node}]:-}" ]; then
    seen_train_nodes[${node}]=1
    train_leaf_jobs+=("${last_train_by_node[${node}]}")
  fi
done
train_dependency="afterok:$(IFS=:; echo "${train_leaf_jobs[*]}")"
eval_jobs=()
declare -A last_eval_by_node=()
for index in "${!nodes[@]}"; do
  if [ -n "${last_eval_by_node[${nodes[$index]}]:-}" ]; then
    eval_dependency="afterok:${last_eval_by_node[${nodes[$index]}]}"
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
  last_eval_by_node[${nodes[$index]}]="${eval_jobs[-1]}"
done

eval_leaf_jobs=()
declare -A seen_eval_nodes=()
for node in "${nodes[@]}"; do
  if [ -z "${seen_eval_nodes[${node}]:-}" ]; then
    seen_eval_nodes[${node}]=1
    eval_leaf_jobs+=("${last_eval_by_node[${node}]}")
  fi
done
eval_dependency="afterok:$(IFS=:; echo "${eval_leaf_jobs[*]}")"
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
