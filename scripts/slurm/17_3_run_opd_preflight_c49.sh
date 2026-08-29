#!/bin/bash
# Run the registered dual-arm signal and concise-prompt adherence preflight.

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

CONFIG="${CONFIG:-configs/capacity_length_opd_prompt_pilot_v1.json}"
RESULT_ROOT="${RESULT_ROOT:-results/capacity_length_opd_prompt_pilot_v1}"
OUTPUT_DIR="${RESULT_ROOT}/pilot/preflight"
SHARD_DIR="${RESULT_ROOT}/pilot/preflight_shards"
if [ "${HOST_NAME}" = "c49" ]; then
  # One dense 7B teacher plus one 1.5B LoRA student peaks below 22 GiB in
  # the registered batch-1 scoring path.  C49 has only 32,760 MiB total and
  # retains the user's ~1.3 GiB gg keepalive, so the A6000 threshold is not
  # physically attainable there.  Require the measured peak plus margin.
  DEFAULT_MIN_FREE_MIB=22000
else
  DEFAULT_MIN_FREE_MIB=30000
fi
MIN_FREE_MIB="${MIN_FREE_MIB:-${DEFAULT_MIN_FREE_MIB}}"
WAIT_SECONDS="${WAIT_SECONDS:-30}"
STABLE_CHECKS="${STABLE_CHECKS:-2}"
RUN_TAG="${RUN_TAG:-${HOST_NAME}_${SLURM_JOB_ID:-allocation}_${SLURM_STEP_ID:-step}}"

conda activate /mnt/beegfs/youyang7/.conda/envs/sft
stage_opd_model_snapshots "${CONFIG}" 1
mkdir_with_retry "${OUTPUT_DIR}"
mkdir_with_retry "${SHARD_DIR}"
python3 -V
nvidia-smi
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true
NUM_SHARDS=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["preflight"]["num_shards"])' "${CONFIG}")
GPU_IDS=()
select_gpus_for_approved_node \
  "${MIN_FREE_MIB}" "${NUM_SHARDS}" "${WAIT_SECONDS}" "${STABLE_CHECKS}"
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
