#!/bin/bash

#SBATCH -J 14_7_paired_rewrite_audit
#SBATCH -p a6000
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/projects/small_language_model/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/small_language_model/logs/%x-%j.err

set -euo pipefail
source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate "${AUDIT_ENV:-fact}"
cd /home/youyang7/projects/small_language_model

python3 -u scripts/14_7_audit_paired_rewrite_completion.py \
  --config "${CONFIG:-configs/capacity_length_paired_rewrite_7b_pilot_v1.json}" \
  --output-root "${OUTPUT_ROOT:-results/capacity_length_paired_rewrite_7b_pilot_v1}" \
  --checkpoint-root "${CHECKPOINT_ROOT:-checkpoints/capacity_length_paired_rewrite_7b_pilot_v1}" \
  --figure-root "${FIGURE_ROOT:-figures/capacity_length_paired_rewrite_7b_pilot_v1}"
