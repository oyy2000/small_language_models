#!/bin/bash

#SBATCH -J 19_2_ranked_teacher_gen
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --oversubscribe
#SBATCH --time=5-00:00:00
#SBATCH --output=/home/%u/projects/small_language_model/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/small_language_model/logs/%x-%j.err

set -euo pipefail

source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
source /home/youyang7/projects/small_language_model/scripts/slurm/_gpu_idle_gate.sh
conda activate "${GENERATION_ENV:-/mnt/beegfs/youyang7/.conda/envs/fact}"
cd /home/youyang7/projects/small_language_model

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/mnt/beegfs/youyang7/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/mnt/beegfs/youyang7/.cache/huggingface/datasets}"

TEACHER_NAME="${TEACHER_NAME:?TEACHER_NAME is required}"
CONFIG="${CONFIG:?CONFIG is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_ranked_sampling_multiteacher_v1}"
SMOKE_DIR="${OUTPUT_ROOT}/smoke/teachers/${TEACHER_NAME}/generation"
GENERATION_DIR="${OUTPUT_ROOT}/formal/teachers/${TEACHER_NAME}/generation"
DATASET_DIR="${OUTPUT_ROOT}/formal/teachers/${TEACHER_NAME}/datasets"
MIN_FREE_MIB="${MIN_FREE_MIB:-18000}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
STABLE_CHECKS="${STABLE_CHECKS:-2}"
MAX_USED_MIB="${MAX_USED_MIB:-500}"
MAX_UTILIZATION="${MAX_UTILIZATION:-10}"
NUM_SHARDS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["generation"]["num_shards"])' "${CONFIG}")

case "${TEACHER_NAME}" in
  qwen2p5_1p5b|qwen2p5_3b|qwen2p5_14b) ;;
  *) echo "This job may generate only the three new teachers: ${TEACHER_NAME}" >&2; exit 2 ;;
esac
if [ "${NUM_SHARDS}" -ne 3 ]; then
  echo "Main-matrix generation requires three shards, observed=${NUM_SHARDS}." >&2
  exit 2
fi
if [ -f "${DATASET_DIR}/GENERATION_COMPLETE" ]; then
  echo "Teacher generation is already complete: ${TEACHER_NAME}" >&2
  exit 2
fi

mkdir_with_retry "${SMOKE_DIR}"
mkdir_with_retry "${GENERATION_DIR}"
echo "host=$(hostname) teacher=${TEACHER_NAME} min_free_mib=${MIN_FREE_MIB}"
nvidia-smi
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true
GPU_IDS=()
select_gpus_for_approved_node \
  "${MIN_FREE_MIB}" "${NUM_SHARDS}" "${WAIT_SECONDS}" "${STABLE_CHECKS}" \
  "${MAX_USED_MIB}" "${MAX_UTILIZATION}"
GPU_IDS=("${GPU_IDS[@]:0:${NUM_SHARDS}}")

smoke_gpu="${GPU_IDS[0]}"
smoke_log="${SMOKE_DIR}/worker_${SLURM_JOB_ID:-manual}_gpu_${smoke_gpu}.log"
CUDA_VISIBLE_DEVICES="${smoke_gpu}" python3 -u scripts/16_1_generate_ranked_length_samples.py \
  --config "${CONFIG}" \
  --output-dir "${SMOKE_DIR}" \
  --num-shards 1 \
  --shard-index 0 \
  --limit 1 \
  --skip-existing >"${smoke_log}" 2>&1
echo "smoke_complete teacher=${TEACHER_NAME} gpu=${smoke_gpu} log=${smoke_log}"

pids=()
for shard_index in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu_id="${GPU_IDS[$shard_index]}"
  worker_log="${GENERATION_DIR}/worker_${SLURM_JOB_ID:-manual}_shard_${shard_index}_gpu_${gpu_id}.log"
  (
    CUDA_VISIBLE_DEVICES="${gpu_id}" python3 -u scripts/16_1_generate_ranked_length_samples.py \
      --config "${CONFIG}" \
      --output-dir "${GENERATION_DIR}" \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "${shard_index}" \
      --skip-existing
  ) >"${worker_log}" 2>&1 &
  pids+=("$!")
  echo "launched teacher=${TEACHER_NAME} shard=${shard_index} gpu=${gpu_id} log=${worker_log}"
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if [ "${failed}" -ne 0 ]; then
  echo "At least one generation shard failed for ${TEACHER_NAME}." >&2
  exit 1
fi

conda activate "${SFT_ENV:-/mnt/beegfs/youyang7/.conda/envs/sft}"
python3 -u scripts/16_2_merge_ranked_length_samples.py \
  --config "${CONFIG}" \
  --input-dir "${GENERATION_DIR}" \
  --output-dir "${DATASET_DIR}"
echo "ranked_teacher_generation_complete teacher=${TEACHER_NAME} output=${DATASET_DIR}"
