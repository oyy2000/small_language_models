#!/bin/bash
# Resume the isolated OPD continuation after the current C49 allocation ends.

set -euo pipefail

cd /home/youyang7/projects/small_language_model

if [ "$(hostname -s | tr '[:upper:]' '[:lower:]')" != "c49" ]; then
  echo "This recovery job is registered for C49." >&2
  exit 2
fi

RUNTIME_ROOT="${LBD_RUNTIME_CHECKPOINT_ROOT:-/var/tmp}"
CONTINUATION_NAME="capacity_length_opd_prompt_gate_waived_continuation_v1"
STAGE="gate_waived_continuation"
standard_state="${RUNTIME_ROOT}/${CONTINUATION_NAME}/${STAGE}/standard_prompt/resume/current_bundle/state.json"
concise_state="${RUNTIME_ROOT}/${CONTINUATION_NAME}/${STAGE}/bounded_concise_prompt/resume/current_bundle/state.json"
CHECKPOINT_ROOT="checkpoints/${CONTINUATION_NAME}/${STAGE}"
standard_marker="${CHECKPOINT_ROOT}/standard_prompt/TRAIN_COMPLETE"
concise_marker="${CHECKPOINT_ROOT}/bounded_concise_prompt/TRAIN_COMPLETE"

if { [ -f "${standard_state}" ] || [ -f "${concise_state}" ]; } \
  && { [ -f "${standard_state}" ] || [ -f "${standard_marker}" ]; } \
  && { [ -f "${concise_state}" ] || [ -f "${concise_marker}" ]; }; then
  full_resume=1
elif [ ! -e "$(dirname "$(dirname "$(dirname "${standard_state}")")")" ] \
  && [ ! -e "$(dirname "$(dirname "$(dirname "${concise_state}")")")" ]; then
  full_resume=0
else
  echo "The two full-training arms do not have a symmetric resumable state; audit before retrying." >&2
  exit 2
fi

exec env SMOKE_RESUME=1 FULL_RESUME="${full_resume}" \
  bash scripts/slurm/17_4_continue_gate_waived_opd_c49.sh
