#!/bin/bash
# Prepare and launch short/medium/long SFT runs on three C49 GPUs.

set -euo pipefail

source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
source /home/youyang7/projects/small_language_model/scripts/slurm/_gpu_idle_gate.sh
conda activate /mnt/beegfs/youyang7/.conda/envs/sft
cd /home/youyang7/projects/small_language_model

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-4}"
export HF_HOME="${HF_HOME:-/mnt/beegfs/youyang7/.cache/huggingface}"

TRAINING_CONFIG="${TRAINING_CONFIG:-configs/capacity_length_ranked_sampling_7b_training_seed17_v1.json}"
TRAINING_OVERLAY="${TRAINING_OVERLAY:-configs/capacity_length_ranked_sampling_7b_sft_seed17_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_ranked_sampling_7b_v1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/capacity_length_ranked_sampling_7b_v1}"
WORK_DIR="${OUTPUT_ROOT}/formal/training"
INPUT_MANIFEST="${WORK_DIR}/input/dataset_manifest.json"
MIN_FREE_MIB="${MIN_FREE_MIB:-18000}"
WAIT_SECONDS="${WAIT_SECONDS:-30}"
STABLE_CHECKS="${STABLE_CHECKS:-2}"
RUN_TAG="${RUN_TAG:-c49_train_${SLURM_JOB_ID}_${SLURM_STEP_ID:-step}}"
LOCAL_RUNTIME_ROOT="${LOCAL_RUNTIME_ROOT:-/var/tmp/${USER}-ranked-length-training/${RUN_TAG}}"

export HF_DATASETS_CACHE="${LOCAL_RUNTIME_ROOT}/hf_datasets"
export TMPDIR="${LOCAL_RUNTIME_ROOT}/tmp"
export LBD_RUNTIME_CHECKPOINT_ROOT="${LOCAL_RUNTIME_ROOT}/checkpoints"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}" "${LBD_RUNTIME_CHECKPOINT_ROOT}"
mkdir_with_retry "${WORK_DIR}/input"
mkdir_with_retry "${CHECKPOINT_ROOT}"

python3 -u scripts/16_3_prepare_ranked_length_training.py \
  --training-config "${TRAINING_CONFIG}" \
  --output-manifest "${INPUT_MANIFEST}" \
  --skip-existing

echo "host=$(hostname) run_tag=${RUN_TAG} min_free_mib=${MIN_FREE_MIB}"
nvidia-smi
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true
GPU_IDS=()
select_stably_memory_fit_gpus "${MIN_FREE_MIB}" 3 "${WAIT_SECONDS}" "${STABLE_CHECKS}"
GPU_IDS=("${GPU_IDS[@]:0:3}")
gpu_csv=$(IFS=,; echo "${GPU_IDS[*]}")
echo "training_gpu_ids=${gpu_csv} runtime_root=${LOCAL_RUNTIME_ROOT}"

python3 -u scripts/6_1_train_capacity_length_students.py \
  --config "${TRAINING_CONFIG}" \
  --training-config "${TRAINING_OVERLAY}" \
  --dataset-manifest "${INPUT_MANIFEST}" \
  --work-dir "${WORK_DIR}" \
  --checkpoint-root "${CHECKPOINT_ROOT}/formal" \
  --gpu-ids "${gpu_csv}" \
  --max-parallel 3 \
  --modes equal_example \
  --skip-complete

python3 -u scripts/16_5_audit_ranked_length_training.py \
  --training-config "${TRAINING_CONFIG}" \
  --prepared-manifest "${INPUT_MANIFEST}" \
  --launch-manifest "${WORK_DIR}/training_manifest_shard_00_of_01.json" \
  --output-dir "${WORK_DIR}/audit"
echo "ranked_length_training_complete output=${WORK_DIR}/audit"
