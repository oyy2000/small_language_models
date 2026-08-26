#!/bin/bash
# Execute the OPD pilot inside an existing C49 grabgpu allocation without closing it.

set -euo pipefail

cd /home/youyang7/projects/small_language_model

ALLOCATION_JOB_ID="${ALLOCATION_JOB_ID:-}"
CONFIG="${CONFIG:-configs/capacity_length_opd_prompt_pilot_v1.json}"
DRY_RUN="${DRY_RUN:-0}"
if [ -z "${ALLOCATION_JOB_ID}" ]; then
  echo "Set ALLOCATION_JOB_ID to the active C49 grabgpu allocation job ID." >&2
  exit 2
fi

stages=(
  scripts/slurm/17_1_generate_opd_references_c49.sh
  scripts/slurm/17_3_run_opd_preflight_c49.sh
  scripts/slurm/17_4_train_opd_policies_c49.sh
  scripts/slurm/17_5_eval_analyze_audit_opd_c49.sh
)

if [ "${DRY_RUN}" = "1" ]; then
  echo "/home/youyang7/miniconda3/bin/python scripts/17_0_prepare_opd_storage.py --config ${CONFIG}"
  for stage in "${stages[@]}"; do
    echo "CONFIG=${CONFIG} srun --jobid=${ALLOCATION_JOB_ID} --overlap --nodes=1 --ntasks=1 bash ${stage}"
  done
  exit 0
fi

allocation_state=$(squeue -h -j "${ALLOCATION_JOB_ID}" -o "%T|%N|%u")
if [ -z "${allocation_state}" ]; then
  echo "Allocation ${ALLOCATION_JOB_ID} is not visible in squeue." >&2
  exit 2
fi
case "${allocation_state}" in
  RUNNING\|*c49*\|"${USER}") ;;
  *)
    echo "Allocation must be RUNNING on C49 and owned by ${USER}; observed=${allocation_state}." >&2
    exit 2
    ;;
esac

export CONFIG
/home/youyang7/miniconda3/bin/python scripts/17_0_prepare_opd_storage.py --config "${CONFIG}"
for stage in "${stages[@]}"; do
  srun --jobid="${ALLOCATION_JOB_ID}" --overlap --nodes=1 --ntasks=1 bash "${stage}"
done
echo "OPD pipeline finished without cancelling or closing allocation ${ALLOCATION_JOB_ID}."
