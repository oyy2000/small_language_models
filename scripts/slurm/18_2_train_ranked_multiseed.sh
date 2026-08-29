#!/bin/bash

#SBATCH -J 18_2_ranked_multiseed_train
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

TRAINING_CONFIG="${TRAINING_CONFIG:-configs/capacity_length_ranked_sampling_7b_training_seed42_73_v1.json}"
TRAINING_OVERLAY="${TRAINING_OVERLAY:-configs/capacity_length_ranked_sampling_7b_sft_seed42_73_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_ranked_sampling_7b_multiseed_v1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/capacity_length_ranked_sampling_7b_multiseed_v1}"
WORK_DIR="${OUTPUT_ROOT}/formal/training/seed42_73"
INPUT_MANIFEST="${WORK_DIR}/input/dataset_manifest.json"
MIN_FREE_MIB="${MIN_FREE_MIB:-18000}"
MIN_GPUS="${MIN_GPUS:-3}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
STABLE_CHECKS="${STABLE_CHECKS:-2}"
MAX_USED_MIB="${MAX_USED_MIB:-500}"
MAX_UTILIZATION="${MAX_UTILIZATION:-10}"
RUN_TAG="${RUN_TAG:-ranked_multiseed_${SLURM_JOB_ID:-manual}}"
LOCAL_RUNTIME_ROOT="${LOCAL_RUNTIME_ROOT:-/var/tmp/${USER}-ranked-multiseed/${RUN_TAG}}"

export HF_DATASETS_CACHE="${LOCAL_RUNTIME_ROOT}/hf_datasets"
export TMPDIR="${LOCAL_RUNTIME_ROOT}/tmp"
export LBD_RUNTIME_CHECKPOINT_ROOT="${LOCAL_RUNTIME_ROOT}/checkpoints"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}" "${LBD_RUNTIME_CHECKPOINT_ROOT}"
mkdir_with_retry "${WORK_DIR}/input"
mkdir_with_retry "${CHECKPOINT_ROOT}"

python3 -u scripts/18_1_prepare_ranked_multiseed_training.py \
  --training-config "${TRAINING_CONFIG}" \
  --output-manifest "${INPUT_MANIFEST}" \
  --skip-existing

echo "host=$(hostname) run_tag=${RUN_TAG} min_free_mib=${MIN_FREE_MIB} min_gpus=${MIN_GPUS}"
nvidia-smi
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true
GPU_IDS=()
select_gpus_for_approved_node \
  "${MIN_FREE_MIB}" "${MIN_GPUS}" "${WAIT_SECONDS}" "${STABLE_CHECKS}" \
  "${MAX_USED_MIB}" "${MAX_UTILIZATION}"
GPU_IDS=("${GPU_IDS[@]:0:${MIN_GPUS}}")
GPU_ID_CSV=$(IFS=,; echo "${GPU_IDS[*]}")
echo "training_gpu_ids=${GPU_ID_CSV} runtime_root=${LOCAL_RUNTIME_ROOT}"

python3 -u scripts/6_1_train_capacity_length_students.py \
  --config "${TRAINING_CONFIG}" \
  --training-config "${TRAINING_OVERLAY}" \
  --dataset-manifest "${INPUT_MANIFEST}" \
  --work-dir "${WORK_DIR}" \
  --checkpoint-root "${CHECKPOINT_ROOT}/formal" \
  --gpu-ids "${GPU_ID_CSV}" \
  --max-parallel "${MIN_GPUS}" \
  --modes equal_example \
  --skip-complete

python3 -u scripts/18_3_audit_ranked_multiseed_training.py \
  --training-config "${TRAINING_CONFIG}" \
  --prepared-manifest "${INPUT_MANIFEST}" \
  --launch-manifest "${WORK_DIR}/training_manifest_shard_00_of_01.json" \
  --output-dir "${WORK_DIR}/audit"

echo "ranked_multiseed_training_complete output=${WORK_DIR}/audit"
