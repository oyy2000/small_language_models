#!/bin/bash

#SBATCH -J 14_1_paired_rewrite_gen
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

CONFIG="${CONFIG:-configs/capacity_length_paired_rewrite_7b_pilot_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_paired_rewrite_7b_pilot_v1}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
MIN_FREE_MIB="${MIN_FREE_MIB:-26000}"
MIN_IDLE_GPUS="${MIN_IDLE_GPUS:-1}"
MAX_USED_MIB="${MAX_USED_MIB:-500}"
MAX_UTILIZATION="${MAX_UTILIZATION:-10}"
STABLE_CHECKS="${STABLE_CHECKS:-2}"
RAW_DIR="${OUTPUT_ROOT}/pilot/raw"

mkdir_with_retry "${RAW_DIR}"
nvidia-smi
GPU_IDS=()
select_stably_idle_gpus \
  "${MIN_FREE_MIB}" "${MIN_IDLE_GPUS}" "${WAIT_SECONDS}" \
  "${MAX_USED_MIB}" "${MAX_UTILIZATION}" "${STABLE_CHECKS}"

pids=()
for shard_index in "${!GPU_IDS[@]}"; do
  gpu_id="${GPU_IDS[$shard_index]}"
  log_path="${RAW_DIR}/worker_${shard_index}_gpu_${gpu_id}.log"
  (
    CUDA_VISIBLE_DEVICES="${gpu_id}" python3 -u scripts/14_1_generate_paired_rewrites.py \
      --config "${CONFIG}" \
      --output-dir "${RAW_DIR}" \
      --num-shards "${#GPU_IDS[@]}" \
      --shard-index "${shard_index}" \
      --skip-complete
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
  echo "At least one paired rewrite generation shard failed." >&2
  exit 1
fi
echo "paired_rewrite_generation_complete shards=${#GPU_IDS[@]} output=${RAW_DIR}"
