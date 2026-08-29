#!/bin/bash

#SBATCH -J 19_5_ranked_matrix_audit
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --oversubscribe
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/projects/small_language_model/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/small_language_model/logs/%x-%j.err

set -euo pipefail
source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate "${SFT_ENV:-/mnt/beegfs/youyang7/.conda/envs/sft}"
cd /home/youyang7/projects/small_language_model

CONFIG="${CONFIG:?CONFIG is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_ranked_sampling_multiteacher_v1}"

python3 -u scripts/19_5_audit_ranked_multiteacher_training.py \
  --config "${CONFIG}" \
  --dataset-manifest "${OUTPUT_ROOT}/formal/sft_data/dataset_manifest.json" \
  --work-dir "${OUTPUT_ROOT}/formal/training" \
  --launcher-shards 3 \
  --output-dir "${OUTPUT_ROOT}/formal/training/audit"
