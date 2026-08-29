#!/bin/bash
# Train the two OPD prompt arms concurrently on separate admitted GPUs.

set -euo pipefail

source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
source /home/youyang7/projects/small_language_model/scripts/slurm/_gpu_idle_gate.sh
cd /home/youyang7/projects/small_language_model

HOST_NAME="$(hostname -s | tr '[:upper:]' '[:lower:]')"
case "${HOST_NAME}" in
  c30|c31|c32|c49) ;;
  *)
    echo "This runner is restricted to the approved nodes C30, C31, C32, and C49." >&2
    exit 2
    ;;
esac

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/mnt/beegfs/youyang7/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/mnt/beegfs/youyang7/.cache/huggingface/datasets}"
export LBD_RUNTIME_CHECKPOINT_ROOT="${LBD_RUNTIME_CHECKPOINT_ROOT:-/var/tmp}"

CONFIG="${CONFIG:-configs/capacity_length_opd_prompt_pilot_v1.json}"
GATE_WAIVER_CONFIG="${GATE_WAIVER_CONFIG:-}"
MAX_PROMPT_BATCHES="${MAX_PROMPT_BATCHES:-}"
DRY_RUN="${DRY_RUN:-0}"
conda activate /mnt/beegfs/youyang7/.conda/envs/sft
if [ -n "${GATE_WAIVER_CONFIG}" ]; then
  mapfile -t waiver_paths < <(
    python3 -c '
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
with path.open(encoding="utf-8") as handle:
    waiver = json.load(handle)
print(waiver["outputs"]["result_root"])
print(waiver["reference_evidence"]["manifest_path"])
print(pathlib.Path(waiver["failed_preflight"]["manifest_path"]).parent)
print(waiver["stages"]["smoke"] if sys.argv[2] else waiver["stages"]["training"])
' "${GATE_WAIVER_CONFIG}" "${MAX_PROMPT_BATCHES}"
  )
  if [ "${#waiver_paths[@]}" -ne 4 ]; then
    echo "Could not resolve the gate-waived continuation paths." >&2
    exit 2
  fi
  RESULT_ROOT="${waiver_paths[0]}"
  REFERENCE_MANIFEST="${waiver_paths[1]}"
  PREFLIGHT_DIR="${waiver_paths[2]}"
  OUTPUT_DIR="${RESULT_ROOT}/${waiver_paths[3]}/training"
else
  if [ -n "${MAX_PROMPT_BATCHES}" ]; then
    echo "MAX_PROMPT_BATCHES requires GATE_WAIVER_CONFIG for this continuation runner." >&2
    exit 2
  fi
  RESULT_ROOT="${RESULT_ROOT:-results/capacity_length_opd_prompt_pilot_v1}"
  REFERENCE_MANIFEST="${RESULT_ROOT}/pilot/references/reference_manifest.json"
  PREFLIGHT_DIR="${RESULT_ROOT}/pilot/preflight"
  OUTPUT_DIR="${RESULT_ROOT}/pilot/training"
fi
if [ "${HOST_NAME}" = "c49" ]; then
  # The registered OPD worker uses a dense 7B teacher, a 1.5B LoRA student,
  # batch-1 token scoring, and gradient checkpointing.  Its expected peak plus
  # safety margin fits within 22 GiB free.  A 32,000 MiB free-memory request is
  # impossible on C49 after retaining the user's gg keepalive.
  DEFAULT_MIN_FREE_MIB=22000
else
  DEFAULT_MIN_FREE_MIB=32000
fi
MIN_FREE_MIB="${MIN_FREE_MIB:-${DEFAULT_MIN_FREE_MIB}}"
WAIT_SECONDS="${WAIT_SECONDS:-30}"
STABLE_CHECKS="${STABLE_CHECKS:-2}"
RESUME="${RESUME:-0}"

stage_opd_model_snapshots "${CONFIG}" 1
mkdir_with_retry "${OUTPUT_DIR}"
python3 -V
nvidia-smi
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true
GPU_IDS=()
select_gpus_for_approved_node "${MIN_FREE_MIB}" 2 "${WAIT_SECONDS}" "${STABLE_CHECKS}"
selected_gpu_ids="${GPU_IDS[0]},${GPU_IDS[1]}"
command=(
  python3 -u scripts/17_4_launch_opd_training.py
  --config "${CONFIG}"
  --reference-manifest "${REFERENCE_MANIFEST}"
  --preflight-dir "${PREFLIGHT_DIR}"
  --gpu-ids "${selected_gpu_ids}"
  --output-dir "${OUTPUT_DIR}"
  --skip-complete
)
if [ "${RESUME}" = "1" ]; then
  command+=(--resume)
fi
if [ -n "${GATE_WAIVER_CONFIG}" ]; then
  command+=(--gate-waiver-config "${GATE_WAIVER_CONFIG}")
fi
if [ -n "${MAX_PROMPT_BATCHES}" ]; then
  command+=(--max-prompt-batches "${MAX_PROMPT_BATCHES}")
fi
if [ "${DRY_RUN}" = "1" ]; then
  command+=(--dry-run)
fi
"${command[@]}"
if [ "${DRY_RUN}" = "1" ]; then
  echo "OPD training launcher dry run complete: ${OUTPUT_DIR}/training_launcher_manifest_dry_run.json"
else
  echo "OPD training launcher complete: ${OUTPUT_DIR}/training_launcher_manifest.json"
fi
