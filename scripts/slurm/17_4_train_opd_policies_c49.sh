#!/bin/bash
# Train the two OPD prompt arms concurrently on separate memory-fit GPUs.

set -euo pipefail

source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
source /home/youyang7/projects/small_language_model/scripts/slurm/_gpu_idle_gate.sh
cd /home/youyang7/projects/small_language_model

if [ "$(hostname -s | tr '[:upper:]' '[:lower:]')" != "c49" ]; then
  echo "This runner is registered for C49 and must execute inside the active C49 allocation." >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/mnt/beegfs/youyang7/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/mnt/beegfs/youyang7/.cache/huggingface/datasets}"
export LBD_RUNTIME_CHECKPOINT_ROOT="${LBD_RUNTIME_CHECKPOINT_ROOT:-/var/tmp}"

CONFIG="${CONFIG:-configs/capacity_length_opd_prompt_pilot_v1.json}"
RESULT_ROOT="${RESULT_ROOT:-results/capacity_length_opd_prompt_pilot_v1}"
REFERENCE_MANIFEST="${RESULT_ROOT}/pilot/references/reference_manifest.json"
PREFLIGHT_DIR="${RESULT_ROOT}/pilot/preflight"
OUTPUT_DIR="${RESULT_ROOT}/pilot/training"
MIN_FREE_MIB="${MIN_FREE_MIB:-32000}"
WAIT_SECONDS="${WAIT_SECONDS:-30}"
STABLE_CHECKS="${STABLE_CHECKS:-2}"
RESUME="${RESUME:-0}"

conda activate /mnt/beegfs/youyang7/.conda/envs/sft
mkdir_with_retry "${OUTPUT_DIR}"
python3 -V
nvidia-smi
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true
GPU_IDS=()
select_stably_memory_fit_gpus "${MIN_FREE_MIB}" 2 "${WAIT_SECONDS}" "${STABLE_CHECKS}"
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
"${command[@]}"
echo "OPD training launcher complete: ${OUTPUT_DIR}/training_launcher_manifest.json"
