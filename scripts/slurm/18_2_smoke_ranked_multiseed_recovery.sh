#!/bin/bash

#SBATCH -J 18_2_ranked_recovery_smoke
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --oversubscribe
#SBATCH --time=06:00:00
#SBATCH --output=/home/%u/projects/small_language_model/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/small_language_model/logs/%x-%j.err

set -euo pipefail

source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
source /home/youyang7/projects/small_language_model/scripts/slurm/_gpu_idle_gate.sh
conda activate "${SFT_ENV:-/mnt/beegfs/youyang7/.conda/envs/sft}"
cd /home/youyang7/projects/small_language_model

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-4}"
export HF_HOME="${HF_HOME:-/mnt/beegfs/youyang7/.cache/huggingface}"

TRAINING_CONFIG="${TRAINING_CONFIG:-configs/capacity_length_ranked_sampling_7b_training_seed42_73_v1.json}"
TRAINING_OVERLAY="${TRAINING_OVERLAY:-configs/capacity_length_ranked_sampling_7b_sft_seed42_73_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_ranked_sampling_7b_multiseed_v1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/capacity_length_ranked_sampling_7b_multiseed_v1}"
INPUT_MANIFEST="${OUTPUT_ROOT}/formal/training/seed42_73/input/dataset_manifest.json"
RECOVERY_DIR="${OUTPUT_ROOT}/recovery/job_${SLURM_JOB_ID}/seed42_short"
RECOVERY_LOCK="${OUTPUT_ROOT}/recovery/seed42_short_recovery.lock"
ADAPTER_MARKER="${CHECKPOINT_ROOT}/formal/equal_example__qwen2p5_7b__relative_short__seed_42/TRAIN_COMPLETE"
MIN_FREE_MIB="${MIN_FREE_MIB:-18000}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
STABLE_CHECKS="${STABLE_CHECKS:-2}"
MAX_USED_MIB="${MAX_USED_MIB:-500}"
MAX_UTILIZATION="${MAX_UTILIZATION:-10}"
LOCAL_RUNTIME_ROOT="${LOCAL_RUNTIME_ROOT:-/var/tmp/${USER}-ranked-multiseed-recovery/job_${SLURM_JOB_ID}}"

for path in "${TRAINING_CONFIG}" "${TRAINING_OVERLAY}" "${INPUT_MANIFEST}"; do
  if [ ! -f "${path}" ]; then
    echo "Required recovery input is missing: ${path}" >&2
    exit 2
  fi
done

export HF_DATASETS_CACHE="${LOCAL_RUNTIME_ROOT}/hf_datasets"
export TMPDIR="${LOCAL_RUNTIME_ROOT}/tmp"
export LBD_RUNTIME_CHECKPOINT_ROOT="${LOCAL_RUNTIME_ROOT}/checkpoints"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}" "${LBD_RUNTIME_CHECKPOINT_ROOT}"
mkdir_with_retry "${RECOVERY_DIR}"

echo "host=$(hostname) recovery_dir=${RECOVERY_DIR} runtime_root=${LOCAL_RUNTIME_ROOT}"
python3 -V
df -h /var/tmp "${HF_HOME}" || true
df -i /var/tmp "${HF_HOME}" || true
nvidia-smi
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true

GPU_IDS=()
select_gpus_for_approved_node \
  "${MIN_FREE_MIB}" 1 "${WAIT_SECONDS}" "${STABLE_CHECKS}" \
  "${MAX_USED_MIB}" "${MAX_UTILIZATION}"

# Several node-pinned copies may race for an idle shared GPU. Take a common
# lock only after a node passes admission, then recheck before training because
# GPU state may have changed while waiting for the lock.
exec 9>"${RECOVERY_LOCK}"
flock 9
if [ ! -f "${ADAPTER_MARKER}" ]; then
  GPU_IDS=()
  select_gpus_for_approved_node \
    "${MIN_FREE_MIB}" 1 "${WAIT_SECONDS}" "${STABLE_CHECKS}" \
    "${MAX_USED_MIB}" "${MAX_UTILIZATION}"
fi
gpu_id="${GPU_IDS[0]:-0}"
echo "recovery_gpu_id=${gpu_id} adapter_marker_exists=$([ -f "${ADAPTER_MARKER}" ] && echo 1 || echo 0)"

# The prepared manifest is ordered as three seed-42 ranks followed by three
# seed-73 ranks. Shard 0 of 6 therefore runs only seed-42 relative-short.
python3 -u scripts/6_1_train_capacity_length_students.py \
  --config "${TRAINING_CONFIG}" \
  --training-config "${TRAINING_OVERLAY}" \
  --dataset-manifest "${INPUT_MANIFEST}" \
  --work-dir "${RECOVERY_DIR}" \
  --checkpoint-root "${CHECKPOINT_ROOT}/formal" \
  --gpu-ids "${gpu_id}" \
  --max-parallel 1 \
  --launcher-shards 6 \
  --launcher-shard-index 0 \
  --modes equal_example \
  --skip-complete

echo "ranked_multiseed_recovery_smoke_complete output=${RECOVERY_DIR}"
