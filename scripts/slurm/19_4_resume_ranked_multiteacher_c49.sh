#!/bin/bash

# Recover one ranked-multiteacher training shard inside the existing C49
# grabgpu allocation, wait for the peer C32 shard, then continue the registered
# training audit and evaluation/analysis/audit pipeline.

set -euo pipefail

PROJECT_ROOT=/home/youyang7/projects/small_language_model
cd "${PROJECT_ROOT}"

CONFIG="${CONFIG:?CONFIG is required}"
TRAINING_OVERLAY="${TRAINING_OVERLAY:?TRAINING_OVERLAY is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_ranked_sampling_multiteacher_v1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/capacity_length_ranked_sampling_multiteacher_v1}"
FIGURE_ROOT="${FIGURE_ROOT:-figures/capacity_length_ranked_sampling_multiteacher_v1}"
PEER_JOB_ID="${PEER_JOB_ID:?PEER_JOB_ID is required}"
PEER_SHARD_INDEX="${PEER_SHARD_INDEX:-2}"
POLL_SECONDS="${POLL_SECONDS:-30}"
PEER_WAIT_LIMIT_SECONDS="${PEER_WAIT_LIMIT_SECONDS:-43200}"

if ! [[ "${PEER_JOB_ID}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid PEER_JOB_ID=${PEER_JOB_ID}." >&2
  exit 2
fi

echo "recovery_stage=train_c49 launcher_shard=0 peer_job=${PEER_JOB_ID}"
env \
  CONFIG="${CONFIG}" \
  TRAINING_OVERLAY="${TRAINING_OVERLAY}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  CHECKPOINT_ROOT="${CHECKPOINT_ROOT}" \
  LAUNCHER_SHARDS=3 \
  LAUNCHER_SHARD_INDEX=0 \
  GPU_ADMISSION_POLICY=memory_fit \
  MIN_FREE_MIB="${TRAIN_MIN_FREE_MIB:-18000}" \
  MIN_GPUS=3 \
  WAIT_SECONDS="${GPU_WAIT_SECONDS:-30}" \
  STABLE_CHECKS="${GPU_STABLE_CHECKS:-2}" \
  RUN_TAG="ranked_matrix_resume_c49_${SLURM_JOB_ID:-manual}_shard_0" \
  bash scripts/slurm/19_4_train_ranked_multiteacher_matrix.sh

peer_manifest="${OUTPUT_ROOT}/formal/training/training_manifest_shard_$(printf '%02d' "${PEER_SHARD_INDEX}")_of_03.json"
waited=0
while true; do
  if python3 - "${peer_manifest}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
payload = json.loads(path.read_text(encoding="utf-8"))
raise SystemExit(0 if payload.get("status") == "complete" else 1)
PY
  then
    break
  fi

  peer_state="$(sacct -X -j "${PEER_JOB_ID}" --noheader --parsable2 --format=State 2>/dev/null | head -n 1 | cut -d'|' -f1 || true)"
  case "${peer_state}" in
    FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*)
      echo "Peer C32 training job failed before its manifest completed: job=${PEER_JOB_ID} state=${peer_state}." >&2
      exit 1
      ;;
  esac
  if [ "${waited}" -ge "${PEER_WAIT_LIMIT_SECONDS}" ]; then
    echo "Timed out waiting for peer training manifest: ${peer_manifest}" >&2
    exit 1
  fi
  echo "recovery_stage=wait_peer job=${PEER_JOB_ID} state=${peer_state:-unknown} waited_seconds=${waited}"
  sleep "${POLL_SECONDS}"
  waited=$((waited + POLL_SECONDS))
done

echo "recovery_stage=training_audit"
env CONFIG="${CONFIG}" OUTPUT_ROOT="${OUTPUT_ROOT}" \
  bash scripts/slurm/19_5_audit_ranked_multiteacher_training.sh

echo "recovery_stage=eval_analysis_audit"
env \
  CONFIG="${CONFIG}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  FIGURE_ROOT="${FIGURE_ROOT}" \
  GPU_ADMISSION_POLICY=memory_fit \
  MIN_FREE_MIB="${EVAL_MIN_FREE_MIB:-16000}" \
  WAIT_SECONDS="${GPU_WAIT_SECONDS:-30}" \
  STABLE_CHECKS="${GPU_STABLE_CHECKS:-2}" \
  RUN_TAG="ranked_matrix_resume_c49_${SLURM_JOB_ID:-manual}_eval" \
  bash scripts/slurm/19_6_eval_analyze_audit_ranked_multiteacher.sh

echo "ranked_multiteacher_recovery_complete output=${OUTPUT_ROOT}"
