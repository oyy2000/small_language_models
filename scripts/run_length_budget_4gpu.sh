#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG="${1:-configs/real_length_budget_template.json}"
OUTPUT_DIR="${2:-results/real_length_budget}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
NUM_SHARDS="${#GPUS[@]}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}/logs"

echo "config=${CONFIG}"
echo "output_dir=${OUTPUT_DIR}"
echo "gpu_ids=${GPU_IDS}"
echo "num_shards=${NUM_SHARDS}"

pids=()
statuses=()

for shard_index in "${!GPUS[@]}"; do
  gpu_id="${GPUS[$shard_index]}"
  log_path="${OUTPUT_DIR}/logs/generate_shard_${shard_index}_gpu_${gpu_id}.log"
  echo "Launching shard ${shard_index}/${NUM_SHARDS} on GPU ${gpu_id}; log=${log_path}"
  (
    CUDA_VISIBLE_DEVICES="${gpu_id}" python3 scripts/1_1_build_length_budget_traces.py \
      --config "${CONFIG}" \
      --output-dir "${OUTPUT_DIR}" \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "${shard_index}"
  ) > "${log_path}" 2>&1 &
  pids+=("$!")
done

for shard_index in "${!pids[@]}"; do
  pid="${pids[$shard_index]}"
  if wait "${pid}"; then
    statuses[$shard_index]=0
    echo "Shard ${shard_index} finished successfully."
  else
    statuses[$shard_index]=1
    echo "Shard ${shard_index} failed. See ${OUTPUT_DIR}/logs/generate_shard_${shard_index}_gpu_${GPUS[$shard_index]}.log" >&2
  fi
done

failed=0
for status in "${statuses[@]}"; do
  if [[ "${status}" != "0" ]]; then
    failed=1
  fi
done

if [[ "${failed}" != "0" ]]; then
  echo "At least one shard failed; skipping merge." >&2
  exit 1
fi

python3 scripts/1_2_merge_trace_shards.py \
  --input-glob "${OUTPUT_DIR}/shard_*.jsonl" \
  --output "${OUTPUT_DIR}/traces_merged.jsonl" \
  --sft-output "${OUTPUT_DIR}/sft_merged.jsonl"

echo "Done."
echo "merged_traces=${OUTPUT_DIR}/traces_merged.jsonl"
echo "merged_sft=${OUTPUT_DIR}/sft_merged.jsonl"
