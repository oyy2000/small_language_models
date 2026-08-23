#!/bin/bash

#SBATCH -J 11_2_logit_kd_analysis
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --oversubscribe
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/projects/small_language_model/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/small_language_model/logs/%x-%j.err

set -euo pipefail
if [ -f /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh ]; then
  source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
else
  source /home/youyang7/miniconda3/etc/profile.d/conda.sh
fi
conda activate "${SFT_ENV:-sft}"
cd /home/youyang7/projects/small_language_model
export MPLCONFIGDIR="${MPLCONFIGDIR:-/var/tmp/${USER}-matplotlib-${SLURM_JOB_ID:-manual}}"
mkdir -p "${MPLCONFIGDIR}"
python3 -u scripts/11_2_analyze_logit_kd_experiment.py \
  --config "${CONFIG:-configs/capacity_length_logit_kd_seed17_v1.json}"
