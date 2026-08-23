#!/bin/bash

#SBATCH -J 11_6_math_mix_analysis
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/projects/small_language_model/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/small_language_model/logs/%x-%j.err

set -euo pipefail
source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate "${ANALYSIS_ENV:-fact}"
cd /home/youyang7/projects/small_language_model

export MPLBACKEND=Agg
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_math_mix_pilot_v1}"
FIGURE_ROOT="${FIGURE_ROOT:-figures/capacity_length_math_mix_pilot_v1}"
python3 -u scripts/11_6_analyze_math_mix_pilot.py \
  --eval-output-dir "${OUTPUT_ROOT}/pilot/eval" \
  --output-dir "${OUTPUT_ROOT}/pilot/analysis" \
  --figure-dir "${FIGURE_ROOT}"
