#!/bin/bash

#SBATCH -J 11_7_math_mix_audit
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/home/%u/projects/small_language_model/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/small_language_model/logs/%x-%j.err

set -euo pipefail
source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate "${AUDIT_ENV:-fact}"
cd /home/youyang7/projects/small_language_model

OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_math_mix_pilot_v1}"
FIGURE_ROOT="${FIGURE_ROOT:-figures/capacity_length_math_mix_pilot_v1}"
python3 -u scripts/11_7_audit_math_mix_pilot.py \
  --output-root "${OUTPUT_ROOT}" \
  --figure-root "${FIGURE_ROOT}"
