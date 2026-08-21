#!/bin/bash

#SBATCH -J 7_1_capacity_length_eval
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --oversubscribe
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/%u/projects/small_language_model/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/small_language_model/logs/%x-%j.err

set -euo pipefail
source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate "${SFT_ENV:-sft}"
source /home/youyang7/projects/small_language_model/scripts/slurm/_gpu_idle_gate.sh
cd /home/youyang7/projects/small_language_model

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/mnt/beegfs/youyang7/.cache/huggingface}"
LOCAL_RUNTIME_ROOT="${LOCAL_RUNTIME_ROOT:-/var/tmp/${USER}-capacity-length-eval/${SLURM_JOB_ID:-manual}}"
export HF_DATASETS_CACHE="${LOCAL_RUNTIME_ROOT}/hf_datasets"
export TMPDIR="${LOCAL_RUNTIME_ROOT}/tmp"

CONFIG="${CONFIG:-configs/capacity_length_factorial_v1.json}"
EVALUATION_CONFIG="${EVALUATION_CONFIG:-configs/capacity_length_factorial_eval_v1.json}"
STAGE="${STAGE:-formal}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_factorial_v1}"
LAUNCHER_SHARDS="${LAUNCHER_SHARDS:-4}"
LAUNCHER_SHARD_INDEX="${LAUNCHER_SHARD_INDEX:?LAUNCHER_SHARD_INDEX is required}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
MIN_FREE_MIB="${MIN_FREE_MIB:-12000}"
MIN_IDLE_GPUS="${MIN_IDLE_GPUS:-1}"
MAX_USED_MIB="${MAX_USED_MIB:-500}"
MAX_UTILIZATION="${MAX_UTILIZATION:-10}"
STABLE_CHECKS="${STABLE_CHECKS:-2}"

mkdir_with_retry "${OUTPUT_ROOT}/${STAGE}/eval/predictions"
mkdir_with_retry "${OUTPUT_ROOT}/${STAGE}/eval/summaries"
mkdir_with_retry "${OUTPUT_ROOT}/${STAGE}/eval/logs"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"

nvidia-smi
GPU_IDS=()
select_stably_idle_gpus \
  "${MIN_FREE_MIB}" "${MIN_IDLE_GPUS}" "${WAIT_SECONDS}" \
  "${MAX_USED_MIB}" "${MAX_UTILIZATION}" "${STABLE_CHECKS}"
GPU_ID_CSV=$(IFS=,; echo "${GPU_IDS[*]}")
python3 -u scripts/7_1_eval_capacity_length_students.py \
  --config "${CONFIG}" \
  --evaluation-config "${EVALUATION_CONFIG}" \
  --training-manifest-glob "${OUTPUT_ROOT}/${STAGE}/training/training_manifest_shard_*.json" \
  --output-dir "${OUTPUT_ROOT}/${STAGE}/eval" \
  --stage "${STAGE}" \
  --gpu-ids "${GPU_ID_CSV}" \
  --launcher-shards "${LAUNCHER_SHARDS}" \
  --launcher-shard-index "${LAUNCHER_SHARD_INDEX}" \
  --skip-complete
