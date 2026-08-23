#!/bin/bash

#SBATCH -J 11_1_math_mix_gen
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --oversubscribe
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/%u/projects/small_language_model/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/small_language_model/logs/%x-%j.err

set -euo pipefail
source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate "${GENERATION_ENV:-fact}"
source /home/youyang7/projects/small_language_model/scripts/slurm/_gpu_idle_gate.sh
cd /home/youyang7/projects/small_language_model

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/mnt/beegfs/youyang7/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/mnt/beegfs/youyang7/.cache/huggingface/datasets}"

CONFIG="${CONFIG:-configs/capacity_length_math_mix_pilot_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_math_mix_pilot_v1}"
GENERATOR_NAME="${GENERATOR_NAME:-qwen2p5_7b}"
LIMIT="${LIMIT:-1000}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
MIN_FREE_MIB="${MIN_FREE_MIB:-20000}"
MIN_IDLE_GPUS="${MIN_IDLE_GPUS:-1}"
MAX_USED_MIB="${MAX_USED_MIB:-500}"
MAX_UTILIZATION="${MAX_UTILIZATION:-10}"
STABLE_CHECKS="${STABLE_CHECKS:-2}"

GPU_IDS=()
nvidia-smi
select_stably_idle_gpus \
  "${MIN_FREE_MIB}" "${MIN_IDLE_GPUS}" "${WAIT_SECONDS}" \
  "${MAX_USED_MIB}" "${MAX_UTILIZATION}" "${STABLE_CHECKS}"

GENERATOR_OUTPUT="${OUTPUT_ROOT}/pilot/raw/${GENERATOR_NAME}"
mkdir_with_retry "${GENERATOR_OUTPUT}"
pids=()
for shard_index in "${!GPU_IDS[@]}"; do
  gpu_id="${GPU_IDS[$shard_index]}"
  log_path="${GENERATOR_OUTPUT}/worker_${shard_index}_gpu_${gpu_id}.log"
  (
    CUDA_VISIBLE_DEVICES="${gpu_id}" python3 -u scripts/5_1_generate_capacity_length_traces.py \
      --config "${CONFIG}" \
      --generator-name "${GENERATOR_NAME}" \
      --output-dir "${GENERATOR_OUTPUT}" \
      --num-shards "${#GPU_IDS[@]}" \
      --shard-index "${shard_index}" \
      --limit "${LIMIT}" \
      --skip-existing
  ) >"${log_path}" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [ "${failed}" -ne 0 ]; then
  echo "At least one MATH generation shard failed." >&2
  exit 1
fi
echo "math_generation_complete gpus=${GPU_IDS[*]} output=${GENERATOR_OUTPUT}"
