#!/bin/bash
# Freeze and submit the full Phase-C generation, training, evaluation, and audit DAG.

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/slurm/_gpu_idle_gate.sh
source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh

TEMPLATE_CONFIG="${TEMPLATE_CONFIG:-configs/capacity_length_ranked_sampling_multiteacher_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_ranked_sampling_multiteacher_v1}"
FIGURE_ROOT="${FIGURE_ROOT:-figures/capacity_length_ranked_sampling_multiteacher_v1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/capacity_length_ranked_sampling_multiteacher_v1}"
BEEGFS_RESULT_ROOT="${BEEGFS_RESULT_ROOT:-/mnt/beegfs/${USER}/projects/small_language_model/results/capacity_length_ranked_sampling_multiteacher_v1}"
BEEGFS_CHECKPOINT_ROOT="${BEEGFS_CHECKPOINT_ROOT:-/mnt/beegfs/${USER}/projects/small_language_model/checkpoints/capacity_length_ranked_sampling_multiteacher_v1}"
PROTOCOL_DIR="${OUTPUT_ROOT}/formal/protocol"
FROZEN_CONFIG="${PROTOCOL_DIR}/frozen_protocol.json"
TRAINING_OVERLAY="${PROTOCOL_DIR}/frozen_sft_overlay.json"
GENERATION_CONFIG_DIR="${PROTOCOL_DIR}/generation_configs"
GENERATION_CONFIG_MANIFEST="${GENERATION_CONFIG_DIR}/generation_config_manifest.json"
LAUNCHER_PLAN="${PROTOCOL_DIR}/launcher_assignment_plan.json"
PHASE_A_MARKER="results/capacity_length_ranked_sampling_7b_multiseed_v1/MULTISEED_COMPLETE"

ensure_beegfs_directory_link() {
  local stable_path="$1"
  local beegfs_path="$2"
  local label="$3"
  local expected_path
  local resolved_path

  mkdir_with_retry "${beegfs_path}"
  expected_path=$(readlink -f "${beegfs_path}")
  if [ -L "${stable_path}" ] || [ -d "${stable_path}" ]; then
    resolved_path=$(readlink -f "${stable_path}")
    if [ "${resolved_path}" != "${expected_path}" ]; then
      echo "${label} path resolves to ${resolved_path}, expected ${expected_path}." >&2
      exit 2
    fi
  elif [ -e "${stable_path}" ]; then
    echo "Refusing non-directory ${label} path: ${stable_path}" >&2
    exit 2
  else
    mkdir_with_retry "$(dirname "${stable_path}")"
    ln -s "${beegfs_path}" "${stable_path}"
  fi
}

if [ ! -f "${PHASE_A_MARKER}" ]; then
  echo "Phase A is not complete; refusing to submit the main matrix: ${PHASE_A_MARKER}" >&2
  exit 2
fi
ensure_beegfs_directory_link "${OUTPUT_ROOT}" "${BEEGFS_RESULT_ROOT}" "Result"
if [ -e "${OUTPUT_ROOT}/MATRIX_COMPLETE" ]; then
  echo "Main matrix is already sealed: ${OUTPUT_ROOT}/MATRIX_COMPLETE" >&2
  exit 2
fi
if [ -e "${FROZEN_CONFIG}" ] || [ -e "${TRAINING_OVERLAY}" ] || [ -e "${GENERATION_CONFIG_DIR}" ] || [ -e "${LAUNCHER_PLAN}" ]; then
  echo "Frozen Phase-C protocol artifacts already exist; audit before resubmission: ${PROTOCOL_DIR}" >&2
  exit 2
fi

mkdir_with_retry "${PROTOCOL_DIR}"
mkdir_with_retry logs
conda activate /mnt/beegfs/youyang7/.conda/envs/sft
python3 -u scripts/19_0_freeze_ranked_multiteacher_protocol.py \
  --template-config "${TEMPLATE_CONFIG}" \
  --output-config "${FROZEN_CONFIG}" \
  --output-training-overlay "${TRAINING_OVERLAY}"
python3 -u scripts/19_1_materialize_ranked_teacher_configs.py \
  --config "${FROZEN_CONFIG}" \
  --output-dir "${GENERATION_CONFIG_DIR}"
python3 -u scripts/19_1_materialize_ranked_launcher_plan.py \
  --config "${FROZEN_CONFIG}" \
  --output "${LAUNCHER_PLAN}"

ensure_beegfs_directory_link "${CHECKPOINT_ROOT}" "${BEEGFS_CHECKPOINT_ROOT}" "Checkpoint"

submit_generation() {
  local partition="$1"
  local node="$2"
  local teacher="$3"
  local min_free="$4"
  sbatch --parsable -p "${partition}" -w "${node}" \
    --export="ALL,TEACHER_NAME=${teacher},CONFIG=${GENERATION_CONFIG_DIR}/${teacher}.json,OUTPUT_ROOT=${OUTPUT_ROOT},MIN_FREE_MIB=${min_free}" \
    scripts/slurm/19_2_generate_ranked_teacher.sh
}

gen_1p5b=$(submit_generation a6000 c31 qwen2p5_1p5b 14000)
gen_3b=$(submit_generation a5000ada c32 qwen2p5_3b 18000)
gen_14b=$(submit_generation a6000 c30 qwen2p5_14b 45000)
generation_dependency="afterok:${gen_1p5b}:${gen_3b}:${gen_14b}"
data_job=$(sbatch --parsable -p a5000ada -w c32 --dependency="${generation_dependency}" \
  --export="ALL,CONFIG=${FROZEN_CONFIG},GENERATION_CONFIG_MANIFEST=${GENERATION_CONFIG_MANIFEST},OUTPUT_ROOT=${OUTPUT_ROOT}" \
  scripts/slurm/19_3_build_ranked_multiteacher_data.sh)

train_jobs=()
train_specs=("a6000 c30 0" "a6000 c31 1" "a5000ada c32 2")
for spec in "${train_specs[@]}"; do
  read -r partition node shard_index <<<"${spec}"
  job=$(sbatch --parsable -p "${partition}" -w "${node}" --dependency="afterok:${data_job}" \
    --export="ALL,CONFIG=${FROZEN_CONFIG},TRAINING_OVERLAY=${TRAINING_OVERLAY},OUTPUT_ROOT=${OUTPUT_ROOT},CHECKPOINT_ROOT=${CHECKPOINT_ROOT},LAUNCHER_SHARDS=3,LAUNCHER_SHARD_INDEX=${shard_index}" \
    scripts/slurm/19_4_train_ranked_multiteacher_matrix.sh)
  train_jobs+=("${job}")
done
training_dependency="afterok:${train_jobs[0]}:${train_jobs[1]}:${train_jobs[2]}"
training_audit_job=$(sbatch --parsable -p a5000ada -w c32 --dependency="${training_dependency}" \
  --export="ALL,CONFIG=${FROZEN_CONFIG},OUTPUT_ROOT=${OUTPUT_ROOT}" \
  scripts/slurm/19_5_audit_ranked_multiteacher_training.sh)
evaluation_job=$(sbatch --parsable -p a6000 -w c31 --dependency="afterok:${training_audit_job}" \
  --export="ALL,CONFIG=${FROZEN_CONFIG},OUTPUT_ROOT=${OUTPUT_ROOT},FIGURE_ROOT=${FIGURE_ROOT}" \
  scripts/slurm/19_6_eval_analyze_audit_ranked_multiteacher.sh)

python3 -u scripts/19_1_record_ranked_multiteacher_submission.py \
  --config "${FROZEN_CONFIG}" \
  --training-overlay "${TRAINING_OVERLAY}" \
  --generation-config-manifest "${GENERATION_CONFIG_MANIFEST}" \
  --launcher-plan "${LAUNCHER_PLAN}" \
  --output "${PROTOCOL_DIR}/submission_manifest.json" \
  --generation-job "qwen2p5_1p5b=${gen_1p5b}" \
  --generation-job "qwen2p5_3b=${gen_3b}" \
  --generation-job "qwen2p5_14b=${gen_14b}" \
  --data-job "${data_job}" \
  --training-job "0=${train_jobs[0]}" \
  --training-job "1=${train_jobs[1]}" \
  --training-job "2=${train_jobs[2]}" \
  --training-audit-job "${training_audit_job}" \
  --evaluation-job "${evaluation_job}" \
  --note "Initial nine-job Phase-C DAG submission."

echo "generation_jobs=${gen_1p5b},${gen_3b},${gen_14b} data_job=${data_job} training_jobs=${train_jobs[*]} training_audit_job=${training_audit_job} evaluation_job=${evaluation_job}"
