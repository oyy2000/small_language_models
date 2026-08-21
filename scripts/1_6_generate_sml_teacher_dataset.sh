#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG="configs/real_length_budget_template.json"
OUTPUT_DIR="results/real_length_budget"
BUDGETS="small,medium,large"
FORCE=0

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/1_6_generate_sml_teacher_dataset.sh [options]

Options:
  --config PATH       Teacher generation config.
                      Default: configs/real_length_budget_template.json
  --output-dir DIR    Output directory for traces and SFT files.
                      Default: results/real_length_budget
  --budgets NAMES     Comma-separated budget names to split from merged JSONL files.
                      Default: small,medium,large
  --force             Remove prior generated files in --output-dir before running.
  -h, --help          Show this message.

Environment:
  GPU_IDS             Comma-separated GPU ids for sharded generation.
                      Default inherited from scripts/run_length_budget_4gpu.sh: 0,1,2,3
  LOG_EVERY           Per-shard progress logging interval.
                      Default inherited from scripts/run_length_budget_4gpu.sh: 10

Outputs:
  <output-dir>/traces_merged.jsonl
  <output-dir>/sft_merged.jsonl
  <output-dir>/traces_small.jsonl
  <output-dir>/traces_medium.jsonl
  <output-dir>/traces_large.jsonl
  <output-dir>/sft_small.jsonl
  <output-dir>/sft_medium.jsonl
  <output-dir>/sft_large.jsonl
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --budgets)
      BUDGETS="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "${PROJECT_ROOT}"

has_existing_outputs() {
  local dir="$1"
  local pattern
  for pattern in \
    "shard_*.jsonl" \
    "sft_shard_*.jsonl" \
    "traces_merged.jsonl" \
    "sft_merged.jsonl" \
    "config_snapshot.json"; do
    if compgen -G "${dir}/${pattern}" > /dev/null; then
      return 0
    fi
  done

  local budget
  IFS=',' read -r -a budget_array <<< "${BUDGETS}"
  for budget in "${budget_array[@]}"; do
    budget="$(echo "${budget}" | xargs)"
    if [[ -n "${budget}" && -e "${dir}/sft_${budget}.jsonl" ]]; then
      return 0
    fi
    if [[ -n "${budget}" && -e "${dir}/traces_${budget}.jsonl" ]]; then
      return 0
    fi
  done
  return 1
}

clean_generated_outputs() {
  local dir="$1"
  mkdir -p "${dir}"
  rm -f \
    "${dir}"/shard_*.jsonl \
    "${dir}"/sft_shard_*.jsonl \
    "${dir}"/traces_merged.jsonl \
    "${dir}"/sft_merged.jsonl \
    "${dir}"/config_snapshot.json

  local budget
  IFS=',' read -r -a budget_array <<< "${BUDGETS}"
  for budget in "${budget_array[@]}"; do
    budget="$(echo "${budget}" | xargs)"
    if [[ -n "${budget}" ]]; then
      rm -f "${dir}/sft_${budget}.jsonl"
      rm -f "${dir}/traces_${budget}.jsonl"
    fi
  done
}

echo "config=${CONFIG}"
echo "output_dir=${OUTPUT_DIR}"
echo "budgets=${BUDGETS}"
echo "gpu_ids=${GPU_IDS:-0,1,2,3}"
echo "log_every=${LOG_EVERY:-10}"
echo "force=${FORCE}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config does not exist: ${CONFIG}" >&2
  exit 1
fi

if [[ -d "${OUTPUT_DIR}" ]] && has_existing_outputs "${OUTPUT_DIR}"; then
  if [[ "${FORCE}" != "1" ]]; then
    echo "Existing generated files found in ${OUTPUT_DIR}." >&2
    echo "Rerun with --force to replace only the known generated trace/SFT files." >&2
    exit 2
  fi
  clean_generated_outputs "${OUTPUT_DIR}"
fi

GPU_IDS="${GPU_IDS:-0,1,2,3}" LOG_EVERY="${LOG_EVERY:-10}" \
  bash scripts/run_length_budget_4gpu.sh "${CONFIG}" "${OUTPUT_DIR}"

python3 scripts/1_5_split_records_by_budget.py \
  --input "${OUTPUT_DIR}/traces_merged.jsonl" \
  --output-dir "${OUTPUT_DIR}" \
  --budgets "${BUDGETS}" \
  --output-prefix traces

python3 scripts/1_5_split_records_by_budget.py \
  --input "${OUTPUT_DIR}/sft_merged.jsonl" \
  --output-dir "${OUTPUT_DIR}" \
  --budgets "${BUDGETS}" \
  --output-prefix sft

echo "Done."
echo "merged_traces=${OUTPUT_DIR}/traces_merged.jsonl"
echo "merged_sft=${OUTPUT_DIR}/sft_merged.jsonl"
IFS=',' read -r -a budget_array <<< "${BUDGETS}"
for budget in "${budget_array[@]}"; do
  budget="$(echo "${budget}" | xargs)"
  if [[ -n "${budget}" ]]; then
    echo "traces_${budget}=${OUTPUT_DIR}/traces_${budget}.jsonl"
    echo "sft_${budget}=${OUTPUT_DIR}/sft_${budget}.jsonl"
  fi
done
