#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

STANDARD_CONFIG="${STANDARD_CONFIG:-configs/standard_prompt_template.json}"
COD_CONFIG="${COD_CONFIG:-configs/chain_of_draft_template.json}"
STANDARD_OUTPUT="${STANDARD_OUTPUT:-results/standard_prompt}"
COD_OUTPUT="${COD_OUTPUT:-results/chain_of_draft}"
GPU_IDS="${GPU_IDS:-2,3}"

cd "${PROJECT_ROOT}"

echo "standard_config=${STANDARD_CONFIG}"
echo "standard_output=${STANDARD_OUTPUT}"
echo "cod_config=${COD_CONFIG}"
echo "cod_output=${COD_OUTPUT}"
echo "gpu_ids=${GPU_IDS}"

GPU_IDS="${GPU_IDS}" bash scripts/run_length_budget_4gpu.sh \
  "${STANDARD_CONFIG}" \
  "${STANDARD_OUTPUT}"

GPU_IDS="${GPU_IDS}" bash scripts/run_length_budget_4gpu.sh \
  "${COD_CONFIG}" \
  "${COD_OUTPUT}"

echo "Done."
echo "standard_sft=${STANDARD_OUTPUT}/sft_merged.jsonl"
echo "cod_sft=${COD_OUTPUT}/sft_merged.jsonl"
