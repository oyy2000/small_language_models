#!/bin/bash

#SBATCH -J 19_4_ranked_matrix_train
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --oversubscribe
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/%u/projects/small_language_model/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/small_language_model/logs/%x-%j.err

set -euo pipefail
source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
source /home/youyang7/projects/small_language_model/scripts/slurm/_gpu_idle_gate.sh
conda activate "${SFT_ENV:-/mnt/beegfs/youyang7/.conda/envs/sft}"
cd /home/youyang7/projects/small_language_model

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-4}"
export HF_HOME="${HF_HOME:-/mnt/beegfs/youyang7/.cache/huggingface}"

CONFIG="${CONFIG:?CONFIG is required}"
TRAINING_OVERLAY="${TRAINING_OVERLAY:?TRAINING_OVERLAY is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_ranked_sampling_multiteacher_v1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/capacity_length_ranked_sampling_multiteacher_v1}"
LAUNCHER_SHARDS="${LAUNCHER_SHARDS:-3}"
LAUNCHER_SHARD_INDEX="${LAUNCHER_SHARD_INDEX:?LAUNCHER_SHARD_INDEX is required}"
MIN_FREE_MIB="${MIN_FREE_MIB:-18000}"
MIN_GPUS="${MIN_GPUS:-3}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
STABLE_CHECKS="${STABLE_CHECKS:-2}"
MAX_USED_MIB="${MAX_USED_MIB:-500}"
MAX_UTILIZATION="${MAX_UTILIZATION:-10}"
RUN_TAG="${RUN_TAG:-ranked_matrix_${SLURM_JOB_ID:-manual}_shard_${LAUNCHER_SHARD_INDEX}}"
LOCAL_RUNTIME_ROOT="${LOCAL_RUNTIME_ROOT:-/var/tmp/${USER}-ranked-matrix/${RUN_TAG}}"

export HF_DATASETS_CACHE="${LOCAL_RUNTIME_ROOT}/hf_datasets"
export TMPDIR="${LOCAL_RUNTIME_ROOT}/tmp"
export LBD_RUNTIME_CHECKPOINT_ROOT="${LOCAL_RUNTIME_ROOT}/checkpoints"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}" "${LBD_RUNTIME_CHECKPOINT_ROOT}"

DATA_MANIFEST="${OUTPUT_ROOT}/formal/sft_data/dataset_manifest.json"
if [ ! -f "${OUTPUT_ROOT}/formal/sft_data/DATA_COMPLETE" ]; then
  echo "Main-matrix data is not complete: ${OUTPUT_ROOT}" >&2
  exit 2
fi

echo "host=$(hostname) launcher_shard=${LAUNCHER_SHARD_INDEX}/${LAUNCHER_SHARDS}"
nvidia-smi
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true
GPU_IDS=()
select_gpus_for_approved_node \
  "${MIN_FREE_MIB}" "${MIN_GPUS}" "${WAIT_SECONDS}" "${STABLE_CHECKS}" \
  "${MAX_USED_MIB}" "${MAX_UTILIZATION}"
GPU_IDS=("${GPU_IDS[@]:0:${MIN_GPUS}}")
GPU_ID_CSV=$(IFS=,; echo "${GPU_IDS[*]}")

python3 -u scripts/6_1_train_capacity_length_students.py \
  --config "${CONFIG}" \
  --training-config "${TRAINING_OVERLAY}" \
  --dataset-manifest "${DATA_MANIFEST}" \
  --work-dir "${OUTPUT_ROOT}/formal/training" \
  --checkpoint-root "${CHECKPOINT_ROOT}/formal" \
  --gpu-ids "${GPU_ID_CSV}" \
  --max-parallel "${MIN_GPUS}" \
  --launcher-shards "${LAUNCHER_SHARDS}" \
  --launcher-shard-index "${LAUNCHER_SHARD_INDEX}" \
  --modes equal_example \
  --skip-complete
