#!/bin/bash

# Evaluate a disjoint tail of the authoritative main-matrix task list on C32.
# The helper writes predictions and summaries only; the C49 launcher remains
# the sole writer of the authoritative evaluation manifest.

#SBATCH -J 19_6_ranked_eval_helper
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
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_ranked_sampling_multiteacher_v1}"
TASK_START_INDEX="${TASK_START_INDEX:-29}"
TASK_END_INDEX="${TASK_END_INDEX:-37}"
MIN_FREE_MIB="${MIN_FREE_MIB:-16000}"
MIN_GPUS="${MIN_GPUS:-2}"
WAIT_SECONDS="${WAIT_SECONDS:-30}"
STABLE_CHECKS="${STABLE_CHECKS:-2}"
RUN_TAG="${RUN_TAG:-ranked_matrix_eval_helper_${SLURM_JOB_ID:-manual}}"
LOCAL_RUNTIME_ROOT="${LOCAL_RUNTIME_ROOT:-/var/tmp/${USER}-ranked-matrix-eval-helper/${RUN_TAG}}"

export HF_DATASETS_CACHE="${LOCAL_RUNTIME_ROOT}/hf_datasets"
export TMPDIR="${LOCAL_RUNTIME_ROOT}/tmp"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"

GPU_IDS=()
select_gpus_for_approved_node \
  "${MIN_FREE_MIB}" "${MIN_GPUS}" "${WAIT_SECONDS}" "${STABLE_CHECKS}"
GPU_IDS=("${GPU_IDS[@]:0:${MIN_GPUS}}")
GPU_ID_CSV=$(IFS=,; echo "${GPU_IDS[*]}")

python3 -u scripts/19_6_eval_ranked_multiteacher_helpers.py \
  --config "${CONFIG}" \
  --eval-manifest "${OUTPUT_ROOT}/formal/eval/eval_manifest_formal_shard_00_of_01.json" \
  --task-start-index "${TASK_START_INDEX}" \
  --task-end-index "${TASK_END_INDEX}" \
  --gpu-ids "${GPU_ID_CSV}" \
  --max-parallel "${MIN_GPUS}" \
  --helper-manifest "${OUTPUT_ROOT}/formal/eval/helpers/c32_tail_manifest.json"
