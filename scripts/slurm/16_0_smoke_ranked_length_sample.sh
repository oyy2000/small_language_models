#!/bin/bash

#SBATCH -J 16_0_ranked_length_smoke
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --oversubscribe
#SBATCH --time=02:00:00
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
SMOKE_DIR="${OUTPUT_ROOT}/smoke/generation"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
MIN_FREE_MIB="${MIN_FREE_MIB:-45000}"
MAX_USED_MIB="${MAX_USED_MIB:-500}"
MAX_UTILIZATION="${MAX_UTILIZATION:-10}"
STABLE_CHECKS="${STABLE_CHECKS:-2}"

mkdir_with_retry "${SMOKE_DIR}"
echo "host=$(hostname) config=${CONFIG} smoke_dir=${SMOKE_DIR}"
python3 -V
nvidia-smi
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true

GPU_IDS=()
select_stably_idle_gpus \
  "${MIN_FREE_MIB}" 1 "${WAIT_SECONDS}" \
  "${MAX_USED_MIB}" "${MAX_UTILIZATION}" "${STABLE_CHECKS}"
gpu_id="${GPU_IDS[0]}"
worker_log="${SMOKE_DIR}/worker_job_${SLURM_JOB_ID}_gpu_${gpu_id}.log"
CUDA_VISIBLE_DEVICES="${gpu_id}" python3 -u scripts/16_1_generate_ranked_length_samples.py \
  --config "${CONFIG}" \
  --output-dir "${SMOKE_DIR}" \
  --num-shards 1 \
  --shard-index 0 \
  --limit 1 \
  --skip-existing >"${worker_log}" 2>&1

echo "ranked_length_smoke_complete gpu=${gpu_id} output=${SMOKE_DIR} log=${worker_log}"
