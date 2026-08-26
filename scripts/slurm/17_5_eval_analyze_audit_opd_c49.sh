#!/bin/bash
# Evaluate both OPD arms and base, analyze paired effects, then run the independent audit.

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

CONFIG="${CONFIG:-configs/capacity_length_opd_prompt_pilot_v1.json}"
RESULT_ROOT="${RESULT_ROOT:-results/capacity_length_opd_prompt_pilot_v1}"
FIGURE_ROOT="${FIGURE_ROOT:-figures/capacity_length_opd_prompt_pilot_v1}"
PRIMARY_DIR="${RESULT_ROOT}/pilot/evaluation/primary"
SECONDARY_DIR="${RESULT_ROOT}/pilot/evaluation/secondary"
ANALYSIS_DIR="${RESULT_ROOT}/pilot/analysis"
FIGURE_DIR="${FIGURE_ROOT}/pilot"
MIN_FREE_MIB="${MIN_FREE_MIB:-12000}"
WAIT_SECONDS="${WAIT_SECONDS:-30}"
STABLE_CHECKS="${STABLE_CHECKS:-2}"

conda activate /mnt/beegfs/youyang7/.conda/envs/sft
mkdir_with_retry "${PRIMARY_DIR}"
mkdir_with_retry "${SECONDARY_DIR}"
python3 -V
nvidia-smi
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true
GPU_IDS=()
select_stably_memory_fit_gpus "${MIN_FREE_MIB}" 3 "${WAIT_SECONDS}" "${STABLE_CHECKS}"
selected_gpu_ids="${GPU_IDS[0]},${GPU_IDS[1]},${GPU_IDS[2]}"

python3 -u scripts/17_5_launch_opd_evaluation.py \
  --config "${CONFIG}" \
  --split-name primary_evaluation \
  --gpu-ids "${selected_gpu_ids}" \
  --max-parallel 3 \
  --output-dir "${PRIMARY_DIR}" \
  --skip-complete

nvidia-smi
GPU_IDS=()
select_stably_memory_fit_gpus "${MIN_FREE_MIB}" 3 "${WAIT_SECONDS}" "${STABLE_CHECKS}"
selected_gpu_ids="${GPU_IDS[0]},${GPU_IDS[1]},${GPU_IDS[2]}"
python3 -u scripts/17_5_launch_opd_evaluation.py \
  --config "${CONFIG}" \
  --split-name secondary_evaluation \
  --gpu-ids "${selected_gpu_ids}" \
  --max-parallel 3 \
  --output-dir "${SECONDARY_DIR}" \
  --skip-complete

python3 -u scripts/17_6_analyze_opd_prompt_pilot.py \
  --config "${CONFIG}" \
  --primary-eval-manifest "${PRIMARY_DIR}/evaluation_launcher_manifest.json" \
  --secondary-eval-manifest "${SECONDARY_DIR}/evaluation_launcher_manifest.json" \
  --output-dir "${ANALYSIS_DIR}" \
  --figure-dir "${FIGURE_DIR}" \
  --skip-complete

python3 -u scripts/17_7_audit_opd_prompt_pilot.py \
  --config "${CONFIG}" \
  --skip-complete
echo "OPD exploratory pilot evaluation, analysis, and audit complete."
