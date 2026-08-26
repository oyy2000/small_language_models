#!/bin/bash
# Generate frozen base-student reference lengths with one independent shard per GPU.

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
SMOKE_DIR="${RESULT_ROOT}/smoke/references"
SHARD_DIR="${RESULT_ROOT}/pilot/reference_shards"
REFERENCE_DIR="${RESULT_ROOT}/pilot/references"
MIN_FREE_MIB="${MIN_FREE_MIB:-12000}"
WAIT_SECONDS="${WAIT_SECONDS:-30}"
STABLE_CHECKS="${STABLE_CHECKS:-2}"
RUN_TAG="${RUN_TAG:-c49_${SLURM_JOB_ID:-allocation}_${SLURM_STEP_ID:-step}}"

conda activate /mnt/beegfs/youyang7/.conda/envs/sft
NUM_SHARDS=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["reference_generation"]["num_shards"])' "${CONFIG}")
if [ "${NUM_SHARDS}" -ne 3 ]; then
  echo "The registered OPD C49 runner requires three reference shards; observed=${NUM_SHARDS}." >&2
  exit 2
fi
mkdir_with_retry "${SMOKE_DIR}"
mkdir_with_retry "${SHARD_DIR}"
python3 -V
nvidia-smi
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true

GPU_IDS=()
select_stably_memory_fit_gpus "${MIN_FREE_MIB}" 1 "${WAIT_SECONDS}" "${STABLE_CHECKS}"
smoke_gpu="${GPU_IDS[0]}"
smoke_log="${SMOKE_DIR}/worker_${RUN_TAG}_gpu_${smoke_gpu}.log"
CUDA_VISIBLE_DEVICES="${smoke_gpu}" python3 -u scripts/17_1_generate_opd_reference_lengths.py \
  --config "${CONFIG}" \
  --output-dir "${SMOKE_DIR}" \
  --num-shards 1 \
  --shard-index 0 \
  --limit 1 \
  --skip-complete >"${smoke_log}" 2>&1
echo "OPD reference smoke complete: gpu=${smoke_gpu} log=${smoke_log}"

GPU_IDS=()
select_stably_memory_fit_gpus "${MIN_FREE_MIB}" "${NUM_SHARDS}" "${WAIT_SECONDS}" "${STABLE_CHECKS}"
GPU_IDS=("${GPU_IDS[@]:0:${NUM_SHARDS}}")
pids=()
for shard_index in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu_id="${GPU_IDS[$shard_index]}"
  worker_log="${SHARD_DIR}/worker_${RUN_TAG}_shard_${shard_index}_gpu_${gpu_id}.log"
  (
    CUDA_VISIBLE_DEVICES="${gpu_id}" python3 -u scripts/17_1_generate_opd_reference_lengths.py \
      --config "${CONFIG}" \
      --output-dir "${SHARD_DIR}" \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "${shard_index}" \
      --skip-complete
  ) >"${worker_log}" 2>&1 &
  pids+=("$!")
  echo "Launched OPD reference shard=${shard_index} gpu=${gpu_id} log=${worker_log}"
done

failed=0
for worker_pid in "${pids[@]}"; do
  if ! wait "${worker_pid}"; then
    failed=1
  fi
done
if [ "${failed}" -ne 0 ]; then
  echo "At least one OPD reference shard failed; merge was not attempted." >&2
  exit 1
fi

python3 -u scripts/17_2_merge_opd_reference_lengths.py \
  --config "${CONFIG}" \
  --input-dir "${SHARD_DIR}" \
  --output-dir "${REFERENCE_DIR}" \
  --skip-complete
echo "OPD frozen references complete: ${REFERENCE_DIR}"
