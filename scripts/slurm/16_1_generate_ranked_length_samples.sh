#!/bin/bash

#SBATCH -J 16_1_ranked_length_gen
#SBATCH -N 1
#SBATCH --cpus-per-task=32
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
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/mnt/beegfs/youyang7/.cache/huggingface/datasets}"

CONFIG="${CONFIG:-configs/capacity_length_ranked_sampling_7b_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_ranked_sampling_7b_v1}"
GENERATION_DIR="${OUTPUT_ROOT}/formal/generation"
REQUIRED_GPUS="${REQUIRED_GPUS:-3}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
MIN_FREE_MIB="${MIN_FREE_MIB:-45000}"
MAX_USED_MIB="${MAX_USED_MIB:-500}"
MAX_UTILIZATION="${MAX_UTILIZATION:-10}"
STABLE_CHECKS="${STABLE_CHECKS:-2}"

config_shards=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["generation"]["num_shards"])' "${CONFIG}")
if [ "${config_shards}" -ne "${REQUIRED_GPUS}" ]; then
  echo "Config num_shards=${config_shards} does not match REQUIRED_GPUS=${REQUIRED_GPUS}." >&2
  exit 2
fi

mkdir_with_retry "${GENERATION_DIR}"
echo "host=$(hostname) config=${CONFIG} generation_dir=${GENERATION_DIR} required_gpus=${REQUIRED_GPUS}"
python3 -V
nvidia-smi
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true

GPU_IDS=()
select_stably_idle_gpus \
  "${MIN_FREE_MIB}" "${REQUIRED_GPUS}" "${WAIT_SECONDS}" \
  "${MAX_USED_MIB}" "${MAX_UTILIZATION}" "${STABLE_CHECKS}"
if [ "${#GPU_IDS[@]}" -lt "${REQUIRED_GPUS}" ]; then
  echo "Stable GPU gate returned ${#GPU_IDS[@]} GPUs, required=${REQUIRED_GPUS}." >&2
  exit 1
fi
GPU_IDS=("${GPU_IDS[@]:0:${REQUIRED_GPUS}}")
echo "generation_gpu_count=${#GPU_IDS[@]} generation_gpu_ids=${GPU_IDS[*]}"

pids=()
for shard_index in $(seq 0 $((REQUIRED_GPUS - 1))); do
  gpu_id="${GPU_IDS[$shard_index]}"
  log_path="${GENERATION_DIR}/worker_job_${SLURM_JOB_ID}_shard_${shard_index}_gpu_${gpu_id}.log"
  (
    CUDA_VISIBLE_DEVICES="${gpu_id}" python3 -u scripts/16_1_generate_ranked_length_samples.py \
      --config "${CONFIG}" \
      --output-dir "${GENERATION_DIR}" \
      --num-shards "${REQUIRED_GPUS}" \
      --shard-index "${shard_index}" \
      --skip-existing
  ) >"${log_path}" 2>&1 &
  worker_pid="$!"
  pids+=("${worker_pid}")
  echo "launched shard=${shard_index} gpu=${gpu_id} pid=${worker_pid} log=${log_path}"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [ "${failed}" -ne 0 ]; then
  echo "At least one ranked-length generation shard failed." >&2
  exit 1
fi
echo "ranked_length_generation_complete shards=${REQUIRED_GPUS} output=${GENERATION_DIR}"
