#!/bin/bash
# Run smoke, three-GPU formal generation, and audited merge on C49.

set -euo pipefail

source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
source /home/youyang7/projects/small_language_model/scripts/slurm/_gpu_idle_gate.sh
cd /home/youyang7/projects/small_language_model

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/mnt/beegfs/youyang7/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/mnt/beegfs/youyang7/.cache/huggingface/datasets}"

CONFIG="${CONFIG:-configs/capacity_length_ranked_sampling_7b_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_ranked_sampling_7b_v1}"
SMOKE_DIR="${OUTPUT_ROOT}/smoke/generation"
GENERATION_DIR="${OUTPUT_ROOT}/formal/generation"
DATASET_DIR="${OUTPUT_ROOT}/formal/datasets"
MIN_FREE_MIB="${MIN_FREE_MIB:-25000}"
WAIT_SECONDS="${WAIT_SECONDS:-30}"
STABLE_CHECKS="${STABLE_CHECKS:-2}"
RUN_TAG="${RUN_TAG:-c49_${SLURM_JOB_ID}_${SLURM_STEP_ID:-step}}"
NUM_SHARDS=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["generation"]["num_shards"])' "${CONFIG}")

config_backend=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["teacher"]["backend"])' "${CONFIG}")
if [ "${config_backend}" != "vllm" ]; then
  echo "C49 runner requires the vllm backend, observed=${config_backend}." >&2
  exit 2
fi
if [ "${NUM_SHARDS}" -ne 3 ]; then
  echo "C49 runner expects three formal shards, observed=${NUM_SHARDS}." >&2
  exit 2
fi

conda activate /mnt/beegfs/youyang7/.conda/envs/fact
mkdir_with_retry "${SMOKE_DIR}"
mkdir_with_retry "${GENERATION_DIR}"
echo "host=$(hostname) run_tag=${RUN_TAG} min_free_mib=${MIN_FREE_MIB}"
python3 -V
nvidia-smi
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true

GPU_IDS=()
select_stably_memory_fit_gpus "${MIN_FREE_MIB}" 1 "${WAIT_SECONDS}" "${STABLE_CHECKS}"
smoke_gpu="${GPU_IDS[0]}"
smoke_log="${SMOKE_DIR}/worker_${RUN_TAG}_gpu_${smoke_gpu}.log"
CUDA_VISIBLE_DEVICES="${smoke_gpu}" python3 -u scripts/16_1_generate_ranked_length_samples.py \
  --config "${CONFIG}" \
  --output-dir "${SMOKE_DIR}" \
  --num-shards 1 \
  --shard-index 0 \
  --limit 1 \
  --skip-existing >"${smoke_log}" 2>&1
echo "smoke_complete gpu=${smoke_gpu} log=${smoke_log}"

GPU_IDS=()
select_stably_memory_fit_gpus \
  "${MIN_FREE_MIB}" "${NUM_SHARDS}" "${WAIT_SECONDS}" "${STABLE_CHECKS}"
GPU_IDS=("${GPU_IDS[@]:0:${NUM_SHARDS}}")
pids=()
for shard_index in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu_id="${GPU_IDS[$shard_index]}"
  worker_log="${GENERATION_DIR}/worker_${RUN_TAG}_shard_${shard_index}_gpu_${gpu_id}.log"
  (
    CUDA_VISIBLE_DEVICES="${gpu_id}" python3 -u scripts/16_1_generate_ranked_length_samples.py \
      --config "${CONFIG}" \
      --output-dir "${GENERATION_DIR}" \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "${shard_index}" \
      --skip-existing
  ) >"${worker_log}" 2>&1 &
  worker_pid="$!"
  pids+=("${worker_pid}")
  echo "launched shard=${shard_index} gpu=${gpu_id} pid=${worker_pid} log=${worker_log}"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [ "${failed}" -ne 0 ]; then
  echo "At least one C49 ranked-length generation shard failed." >&2
  exit 1
fi

conda activate /mnt/beegfs/youyang7/.conda/envs/sft
python3 -u scripts/16_2_merge_ranked_length_samples.py \
  --config "${CONFIG}" \
  --input-dir "${GENERATION_DIR}" \
  --output-dir "${DATASET_DIR}"
echo "ranked_length_c49_pipeline_complete output=${DATASET_DIR}"
