#!/bin/bash
# Submit the complete validation-to-formal logit-KD dependency graph.

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/capacity_length_logit_kd_seed17_v1.json}"
DRY_RUN="${DRY_RUN:-0}"
nodes=(c32 c49)
partitions=(a5000ada a5000ada)
launcher_shards="${#nodes[@]}"

if [ "${DRY_RUN}" = "1" ]; then
  echo "sbatch --parsable -p ${partitions[0]} -w ${nodes[0]} --export=ALL,CONFIG=${CONFIG} scripts/slurm/9_0_prepare_logit_kd_experiment.sh"
  prepare_job="DRY_PREPARE"
else
  prepare_job=$(sbatch --parsable -p "${partitions[0]}" -w "${nodes[0]}" \
    --export="ALL,CONFIG=${CONFIG}" scripts/slurm/9_0_prepare_logit_kd_experiment.sh)
fi

smoke_jobs=()
if [ "${DRY_RUN}" = "1" ]; then
  echo "sbatch --dependency=afterok:${prepare_job} --export=ALL,CONFIG=${CONFIG},SMOKE_LABEL=a5000ada scripts/slurm/9_3_smoke_logit_kd_gpu.sh"
  smoke_jobs+=("DRY_SMOKE_0")
else
  smoke_jobs+=("$(sbatch --parsable -p "${partitions[0]}" -w "${nodes[0]}" \
    --dependency="afterok:${prepare_job}" --export="ALL,CONFIG=${CONFIG},SMOKE_LABEL=a5000ada" \
    scripts/slurm/9_3_smoke_logit_kd_gpu.sh)")
fi
smoke_dependency="afterok:$(IFS=:; echo "${smoke_jobs[*]}")"

validation_train_jobs=()
for index in "${!nodes[@]}"; do
  command=(
    sbatch --parsable -p "${partitions[$index]}" -w "${nodes[$index]}"
    --dependency="${smoke_dependency}"
    --export="ALL,CONFIG=${CONFIG},STAGE=validation,LAUNCHER_SHARDS=${launcher_shards},LAUNCHER_SHARD_INDEX=${index}"
    scripts/slurm/9_2_train_logit_kd_stage.sh
  )
  if [ "${DRY_RUN}" = "1" ]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    validation_train_jobs+=("DRY_VALIDATION_TRAIN_${index}")
  else
    validation_train_jobs+=("$("${command[@]}")")
  fi
done
validation_train_dependency="afterok:$(IFS=:; echo "${validation_train_jobs[*]}")"

validation_eval_jobs=()
for index in "${!nodes[@]}"; do
  command=(
    sbatch --parsable -p "${partitions[$index]}" -w "${nodes[$index]}"
    --dependency="${validation_train_dependency}"
    --export="ALL,CONFIG=${CONFIG},STAGE=validation,LAUNCHER_SHARDS=${launcher_shards},LAUNCHER_SHARD_INDEX=${index}"
    scripts/slurm/10_0_eval_logit_kd_stage.sh
  )
  if [ "${DRY_RUN}" = "1" ]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    validation_eval_jobs+=("DRY_VALIDATION_EVAL_${index}")
  else
    validation_eval_jobs+=("$("${command[@]}")")
  fi
done
validation_eval_dependency="afterok:$(IFS=:; echo "${validation_eval_jobs[*]}")"
if [ "${DRY_RUN}" = "1" ]; then
  selection_job="DRY_SELECTION"
  echo "sbatch --dependency=${validation_eval_dependency} scripts/slurm/10_2_select_logit_kd_hparams.sh"
else
  selection_job=$(sbatch --parsable -p "${partitions[0]}" -w "${nodes[0]}" \
    --dependency="${validation_eval_dependency}" --export="ALL,CONFIG=${CONFIG}" \
    scripts/slurm/10_2_select_logit_kd_hparams.sh)
fi

formal_train_jobs=()
for index in "${!nodes[@]}"; do
  command=(
    sbatch --parsable -p "${partitions[$index]}" -w "${nodes[$index]}"
    --dependency="afterok:${selection_job}"
    --export="ALL,CONFIG=${CONFIG},STAGE=formal,LAUNCHER_SHARDS=${launcher_shards},LAUNCHER_SHARD_INDEX=${index}"
    scripts/slurm/9_2_train_logit_kd_stage.sh
  )
  if [ "${DRY_RUN}" = "1" ]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    formal_train_jobs+=("DRY_FORMAL_TRAIN_${index}")
  else
    formal_train_jobs+=("$("${command[@]}")")
  fi
done
formal_train_dependency="afterok:$(IFS=:; echo "${formal_train_jobs[*]}")"

formal_eval_jobs=()
for index in "${!nodes[@]}"; do
  command=(
    sbatch --parsable -p "${partitions[$index]}" -w "${nodes[$index]}"
    --dependency="${formal_train_dependency}"
    --export="ALL,CONFIG=${CONFIG},STAGE=formal,LAUNCHER_SHARDS=${launcher_shards},LAUNCHER_SHARD_INDEX=${index}"
    scripts/slurm/10_0_eval_logit_kd_stage.sh
  )
  if [ "${DRY_RUN}" = "1" ]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    formal_eval_jobs+=("DRY_FORMAL_EVAL_${index}")
  else
    formal_eval_jobs+=("$("${command[@]}")")
  fi
done
formal_eval_dependency="afterok:$(IFS=:; echo "${formal_eval_jobs[*]}")"

logit_jobs=()
for index in "${!nodes[@]}"; do
  command=(
    sbatch --parsable -p "${partitions[$index]}" -w "${nodes[$index]}"
    --dependency="${formal_eval_dependency}"
    --export="ALL,CONFIG=${CONFIG},LAUNCHER_SHARDS=${launcher_shards},LAUNCHER_SHARD_INDEX=${index}"
    scripts/slurm/11_0_extract_matched_logit_snapshots.sh
  )
  if [ "${DRY_RUN}" = "1" ]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    logit_jobs+=("DRY_LOGITS_${index}")
  else
    logit_jobs+=("$("${command[@]}")")
  fi
done
logit_dependency="afterok:$(IFS=:; echo "${logit_jobs[*]}")"
if [ "${DRY_RUN}" = "1" ]; then
  analysis_job="DRY_ANALYSIS"
  echo "sbatch --dependency=${logit_dependency} scripts/slurm/11_2_analyze_logit_kd_experiment.sh"
  echo "sbatch --dependency=afterok:${analysis_job} scripts/slurm/12_1_audit_logit_kd_completion.sh"
else
  analysis_job=$(sbatch --parsable -p "${partitions[0]}" -w "${nodes[0]}" \
    --dependency="${logit_dependency}" --export="ALL,CONFIG=${CONFIG}" \
    scripts/slurm/11_2_analyze_logit_kd_experiment.sh)
  audit_job=$(sbatch --parsable -p "${partitions[0]}" -w "${nodes[0]}" \
    --dependency="afterok:${analysis_job}" --export="ALL,CONFIG=${CONFIG}" \
    scripts/slurm/12_1_audit_logit_kd_completion.sh)
  echo "prepare_job=${prepare_job} smoke_jobs=$(IFS=,; echo "${smoke_jobs[*]}") validation_train_jobs=$(IFS=,; echo "${validation_train_jobs[*]}") validation_eval_jobs=$(IFS=,; echo "${validation_eval_jobs[*]}") selection_job=${selection_job} formal_train_jobs=$(IFS=,; echo "${formal_train_jobs[*]}") formal_eval_jobs=$(IFS=,; echo "${formal_eval_jobs[*]}") logit_jobs=$(IFS=,; echo "${logit_jobs[*]}") analysis_job=${analysis_job} audit_job=${audit_job}"
fi
