#!/bin/bash
# Resume the paired-rewrite pipeline after an independently launched selection evaluation.

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/capacity_length_paired_rewrite_7b_pilot_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_paired_rewrite_7b_pilot_v1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/capacity_length_paired_rewrite_7b_pilot_v1}"
FIGURE_ROOT="${FIGURE_ROOT:-figures/capacity_length_paired_rewrite_7b_pilot_v1}"
SELECTION_MANIFEST="${SELECTION_MANIFEST:-${OUTPUT_ROOT}/pilot/eval/selection/eval_manifest_selection_shard_00_of_01.json}"
POLL_SECONDS="${POLL_SECONDS:-30}"

SFT_ENV="${SFT_ENV:-/mnt/beegfs/youyang7/.conda/envs/sft}"
FACT_ENV="${FACT_ENV:-/mnt/beegfs/youyang7/.conda/envs/fact}"
MIN_FREE_MIB="${MIN_FREE_MIB:-12000}"
MAX_USED_MIB="${MAX_USED_MIB:-500}"
MAX_UTILIZATION="${MAX_UTILIZATION:-10}"
STABLE_CHECKS="${STABLE_CHECKS:-2}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
LOCAL_RUNTIME_ROOT="${LOCAL_RUNTIME_ROOT:-/var/tmp/${USER}-paired-rewrite-resume/${SLURM_JOB_ID:-manual}}"

export CONFIG OUTPUT_ROOT CHECKPOINT_ROOT FIGURE_ROOT
export SFT_ENV
export EVAL_ENV="${SFT_ENV}"
export ANALYSIS_ENV="${FACT_ENV}"
export AUDIT_ENV="${FACT_ENV}"
export MIN_FREE_MIB MAX_USED_MIB MAX_UTILIZATION STABLE_CHECKS WAIT_SECONDS
export LOCAL_RUNTIME_ROOT

if ! [[ "${POLL_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "POLL_SECONDS must be a positive integer, got ${POLL_SECONDS}." >&2
  exit 2
fi

echo "Waiting for selection evaluation manifest: ${SELECTION_MANIFEST}"
while true; do
  if [ -f "${SELECTION_MANIFEST}" ]; then
    selection_status=$(jq -r '.status // "missing"' "${SELECTION_MANIFEST}")
    case "${selection_status}" in
      complete)
        echo "Selection evaluation is complete."
        break
        ;;
      failed)
        echo "Selection evaluation failed; refusing to run dependent stages." >&2
        exit 1
        ;;
      prepared|running|missing)
        ;;
      *)
        echo "Unexpected selection evaluation status: ${selection_status}" >&2
        exit 1
        ;;
    esac
  fi
  sleep "${POLL_SECONDS}"
done

echo "Starting selection analysis."
ANALYSIS_STAGE=selection bash scripts/slurm/14_6_analyze_paired_rewrites.sh

echo "Starting final paired-rewrite training."
TRAINING_STAGE=final bash scripts/slurm/14_4_train_paired_rewrites.sh

echo "Starting confirmatory evaluation."
EVAL_STAGE=confirmatory bash scripts/slurm/14_5_eval_paired_rewrites.sh

echo "Starting final analysis."
ANALYSIS_STAGE=final bash scripts/slurm/14_6_analyze_paired_rewrites.sh

echo "Starting completion audit."
bash scripts/slurm/14_7_audit_paired_rewrites.sh

echo "Paired-rewrite resume pipeline completed."
