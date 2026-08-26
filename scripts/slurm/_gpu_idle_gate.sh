#!/bin/bash

# Internal helper sourced by GPU-heavy Slurm entrypoints. The cluster does not
# expose reliable GPU GRES isolation, so only GPUs that remain idle across
# multiple observations are returned to the caller.

mkdir_with_retry() {
  local target="${1:?directory path is required}"
  local attempts="${2:-10}"
  local wait_seconds="${3:-10}"
  local attempt
  for ((attempt = 1; attempt <= attempts; attempt++)); do
    if mkdir -p "${target}"; then
      return 0
    fi
    echo "Directory creation attempt ${attempt}/${attempts} failed for ${target}; retrying." >&2
    sleep "${wait_seconds}"
  done
  echo "Could not create directory after ${attempts} attempts: ${target}" >&2
  return 1
}

select_stably_idle_gpus() {
  local min_free_mib="${1:?minimum free memory is required}"
  local min_idle_gpus="${2:-1}"
  local wait_seconds="${3:-30}"
  local max_used_mib="${4:-500}"
  local max_utilization="${5:-10}"
  local stable_checks="${6:-2}"
  local stable_count=0
  local previous_signature=""
  local signature=""
  local gpu_id used_mib free_mib utilization
  local -a all_gpu_ids=()
  local -a idle_gpu_ids=()

  if [ "${min_idle_gpus}" -le 0 ] || [ "${stable_checks}" -le 0 ]; then
    echo "min_idle_gpus and stable_checks must be positive." >&2
    return 2
  fi
  mapfile -t all_gpu_ids < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
  if [ "${#all_gpu_ids[@]}" -eq 0 ]; then
    echo "No GPUs detected on $(hostname)." >&2
    return 1
  fi

  while true; do
    idle_gpu_ids=()
    for gpu_id in "${all_gpu_ids[@]}"; do
      IFS=',' read -r used_mib free_mib utilization < <(
        nvidia-smi -i "${gpu_id}" \
          --query-gpu=memory.used,memory.free,utilization.gpu \
          --format=csv,noheader,nounits
      )
      used_mib="${used_mib//[[:space:]]/}"
      free_mib="${free_mib//[[:space:]]/}"
      utilization="${utilization//[[:space:]]/}"
      if [ "${free_mib}" -ge "${min_free_mib}" ] \
        && [ "${used_mib}" -le "${max_used_mib}" ] \
        && [ "${utilization}" -le "${max_utilization}" ]; then
        idle_gpu_ids+=("${gpu_id}")
      fi
    done

    signature="${idle_gpu_ids[*]}"
    if [ "${#idle_gpu_ids[@]}" -ge "${min_idle_gpus}" ]; then
      if [ "${signature}" = "${previous_signature}" ]; then
        stable_count=$((stable_count + 1))
      else
        stable_count=1
      fi
      previous_signature="${signature}"
      if [ "${stable_count}" -ge "${stable_checks}" ]; then
        GPU_IDS=("${idle_gpu_ids[@]}")
        echo "Selected stable idle GPUs on $(hostname): ${GPU_IDS[*]}"
        return 0
      fi
    else
      stable_count=0
      previous_signature=""
    fi

    echo "Waiting without interference: idle=${#idle_gpu_ids[@]}/${#all_gpu_ids[@]}, required>=${min_idle_gpus}, stable_checks=${stable_count}/${stable_checks}, free>=${min_free_mib} MiB, used<=${max_used_mib} MiB, utilization<=${max_utilization}%."
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
    sleep "${wait_seconds}"
  done
}

# Select GPUs by remaining capacity only. This is intended for C49, where the
# user's grabgpu keepalive processes are expected and utilization is not an
# admission criterion. Existing process memory is already reflected in
# memory.free; callers must set a threshold that includes workload headroom.
select_stably_memory_fit_gpus() {
  local min_free_mib="${1:?minimum free memory is required}"
  local min_gpus="${2:-1}"
  local wait_seconds="${3:-30}"
  local stable_checks="${4:-2}"
  local stable_count=0
  local previous_signature=""
  local signature=""
  local gpu_id free_mib
  local -a eligible_gpu_ids=()

  if [ "${min_gpus}" -le 0 ] || [ "${stable_checks}" -le 0 ]; then
    echo "min_gpus and stable_checks must be positive." >&2
    return 2
  fi

  while true; do
    eligible_gpu_ids=()
    while IFS=',' read -r gpu_id free_mib; do
      gpu_id="${gpu_id//[[:space:]]/}"
      free_mib="${free_mib//[[:space:]]/}"
      if [ "${free_mib}" -ge "${min_free_mib}" ]; then
        eligible_gpu_ids+=("${gpu_id}")
      fi
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)

    signature="${eligible_gpu_ids[*]}"
    if [ "${#eligible_gpu_ids[@]}" -ge "${min_gpus}" ]; then
      if [ "${signature}" = "${previous_signature}" ]; then
        stable_count=$((stable_count + 1))
      else
        stable_count=1
      fi
      previous_signature="${signature}"
      if [ "${stable_count}" -ge "${stable_checks}" ]; then
        GPU_IDS=("${eligible_gpu_ids[@]}")
        echo "Selected stable memory-fit GPUs on $(hostname): ${GPU_IDS[*]}"
        return 0
      fi
    else
      stable_count=0
      previous_signature=""
    fi

    echo "Waiting for memory capacity without interference: eligible=${#eligible_gpu_ids[@]}/${min_gpus}, stable_checks=${stable_count}/${stable_checks}, free>=${min_free_mib} MiB."
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true
    sleep "${wait_seconds}"
  done
}
