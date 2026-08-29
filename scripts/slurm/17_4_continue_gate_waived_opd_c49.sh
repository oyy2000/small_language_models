#!/bin/bash
# Run the isolated gate-waived OPD smoke, then continue to the full two-arm training.

set -euo pipefail

cd /home/youyang7/projects/small_language_model

if [ "$(hostname -s | tr '[:upper:]' '[:lower:]')" != "c49" ]; then
  echo "This continuation must run inside the user's active C49 allocation." >&2
  exit 2
fi

CONFIG="${CONFIG:-configs/capacity_length_opd_prompt_pilot_v1.json}"
GATE_WAIVER_CONFIG="${GATE_WAIVER_CONFIG:-configs/capacity_length_opd_prompt_gate_waived_continuation_v1.json}"
SMOKE_RESUME="${SMOKE_RESUME:-0}"
FULL_RESUME="${FULL_RESUME:-0}"

echo "Starting isolated gate-waived OPD smoke on C49."
CONFIG="${CONFIG}" \
GATE_WAIVER_CONFIG="${GATE_WAIVER_CONFIG}" \
MAX_PROMPT_BATCHES=1 \
RESUME="${SMOKE_RESUME}" \
bash scripts/slurm/17_4_train_opd_policies_c49.sh

echo "Gate-waived smoke completed; starting the isolated full continuation."
CONFIG="${CONFIG}" \
GATE_WAIVER_CONFIG="${GATE_WAIVER_CONFIG}" \
RESUME="${FULL_RESUME}" \
bash scripts/slurm/17_4_train_opd_policies_c49.sh
