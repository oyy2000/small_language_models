#!/bin/bash
# Run the registered dual-arm signal and concise-prompt adherence preflight.

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
OUTPUT_DIR="${RESULT_ROOT}/pilot/preflight"
SHARD_DIR="${RESULT_ROOT}/pilot/preflight_shards"
MIN_FREE_MIB="${MIN_FREE_MIB:-30000}"
WAIT_SECONDS="${WAIT_SECONDS:-30}"
STABLE_CHECKS="${STABLE_CHECKS:-2}"
RUN_TAG="${RUN_TAG:-c49_${SLURM_JOB_ID:-allocation}_${SLURM_STEP_ID:-step}}"

conda activate /mnt/beegfs/youyang7/.conda/envs/sft
mkdir_with_retry "${OUTPUT_DIR}"
mkdir_with_retry "${SHARD_DIR}"
python3 -V
nvidia-smi
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true
NUM_SHARDS=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["preflight"]["num_shards"])' "${CONFIG}")
GPU_IDS=()
select_stably_memory_fit_gpus "${MIN_FREE_MIB}" "${NUM_SHARDS}" "${WAIT_SECONDS}" "${STABLE_CHECKS}"
GPU_IDS=("${GPU_IDS[@]:0:${NUM_SHARDS}}")
pids=()
for shard_index in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu_id="${GPU_IDS[$shard_index]}"
  log_path="${SHARD_DIR}/preflight_${RUN_TAG}_shard_${shard_index}_gpu_${gpu_id}.log"
  (
    CUDA_VISIBLE_DEVICES="${gpu_id}" python3 -u scripts/17_3_preflight_opd_signal.py \
      --config "${CONFIG}" \
      --output-dir "${SHARD_DIR}" \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "${shard_index}" \
      --skip-complete
  ) >"${log_path}" 2>&1 &
  pids+=("$!")
  echo "Launched OPD preflight shard=${shard_index} gpu=${gpu_id} log=${log_path}"
done

failed=0
for worker_pid in "${pids[@]}"; do
  if ! wait "${worker_pid}"; then
    failed=1
  fi
done
if [ "${failed}" -ne 0 ]; then
  echo "At least one OPD preflight shard failed; global gate was not applied." >&2
  exit 1
fi

python3 -u scripts/17_3_merge_opd_preflight.py \
  --config "${CONFIG}" \
  --input-dir "${SHARD_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --skip-complete
echo "OPD global preflight passed: ${OUTPUT_DIR}/PREFLIGHT_COMPLETE"
