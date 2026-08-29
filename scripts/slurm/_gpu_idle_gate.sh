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

# Apply the project admission policy for the approved GPU nodes.  The default
# remains node-specific (capacity-only on C49, strict idle checks elsewhere).
# A recovery launcher may explicitly set GPU_ADMISSION_POLICY=memory_fit after
# a physical process inventory confirms that remaining memory plus the chosen
# safety threshold is sufficient.  This opt-in avoids weakening every caller.
select_gpus_for_approved_node() {
  local min_free_mib="${1:?minimum free memory is required}"
  local min_gpus="${2:-1}"
  local wait_seconds="${3:-30}"
  local stable_checks="${4:-2}"
  local max_used_mib="${5:-500}"
  local max_utilization="${6:-10}"
  local host
  local admission_policy="${GPU_ADMISSION_POLICY:-auto}"
  host="$(hostname -s | tr '[:upper:]' '[:lower:]')"

  case "${host}" in
    c30|c31|c32|c49)
      ;;
    *)
      echo "GPU admission is not registered for node ${host}." >&2
      return 2
      ;;
  esac

  case "${admission_policy}" in
    auto)
      if [ "${host}" = "c49" ]; then
        select_stably_memory_fit_gpus \
          "${min_free_mib}" "${min_gpus}" "${wait_seconds}" "${stable_checks}"
      else
        select_stably_idle_gpus \
          "${min_free_mib}" "${min_gpus}" "${wait_seconds}" \
          "${max_used_mib}" "${max_utilization}" "${stable_checks}"
      fi
      ;;
    memory_fit)
      select_stably_memory_fit_gpus \
        "${min_free_mib}" "${min_gpus}" "${wait_seconds}" "${stable_checks}"
      ;;
    idle)
      select_stably_idle_gpus \
        "${min_free_mib}" "${min_gpus}" "${wait_seconds}" \
        "${max_used_mib}" "${max_utilization}" "${stable_checks}"
      ;;
    *)
      echo "Unknown GPU_ADMISSION_POLICY=${admission_policy}; expected auto, idle, or memory_fit." >&2
      return 2
      ;;
  esac
}

# Copy an immutable Hugging Face snapshot from the shared cache to node-local
# storage. This avoids mmap faults against large Safetensors files on BeeGFS.
stage_hf_snapshot_to_local() {
  local model_name="${1:?model name is required}"
  local revision="${2:?model revision is required}"
  local hub_cache="${HF_HUB_CACHE:?HF_HUB_CACHE must be set}"
  local local_root="${LBD_LOCAL_MODEL_ROOT:-/var/tmp/${USER}/hf_snapshots}"
  local model_key="${model_name//\//--}"
  local source_dir="${hub_cache}/models--${model_key}/snapshots/${revision}"
  local destination="${local_root}/models--${model_key}/snapshots/${revision}"
  local marker="${destination}/.lbd_snapshot_complete"
  local lock_path="${destination}.lock"
  local source_signature destination_signature

  if [ ! -d "${source_dir}" ]; then
    echo "Registered Hugging Face snapshot is missing: ${source_dir}" >&2
    return 2
  fi
  mkdir_with_retry "$(dirname "${destination}")"
  exec {snapshot_lock_fd}>"${lock_path}"
  flock "${snapshot_lock_fd}"
  source_signature="$({
    find -L "${source_dir}" -maxdepth 1 -type f -printf '%f|%s\n'
  } | sort | sha256sum | awk '{print $1}')"
  if [ ! -f "${marker}" ] || [ "$(<"${marker}")" != "${source_signature}" ]; then
    mkdir_with_retry "${destination}"
    rsync -aL --partial "${source_dir}/" "${destination}/"
    destination_signature="$({
      find "${destination}" -maxdepth 1 -type f ! -name '.lbd_snapshot_complete' -printf '%f|%s\n'
    } | sort | sha256sum | awk '{print $1}')"
    if [ "${destination_signature}" != "${source_signature}" ]; then
      echo "Node-local model snapshot validation failed: ${destination}" >&2
      return 1
    fi
    printf '%s\n' "${source_signature}" >"${marker}"
  fi
  flock -u "${snapshot_lock_fd}"
  exec {snapshot_lock_fd}>&-
  STAGED_HF_SNAPSHOT="${destination}"
  echo "Validated node-local model snapshot: ${STAGED_HF_SNAPSHOT}"
}

stage_opd_model_snapshots() {
  local config_path="${1:?OPD config path is required}"
  local include_teacher="${2:-0}"
  local student_name student_revision teacher_name teacher_revision

  IFS='|' read -r student_name student_revision < <(
    python3 -c 'import json, sys; c=json.load(open(sys.argv[1])); m=c["models"]["student"]; print(m["model_name"] + "|" + m["revision"])' "${config_path}"
  )
  stage_hf_snapshot_to_local "${student_name}" "${student_revision}"
  export LBD_STUDENT_MODEL_SOURCE="${STAGED_HF_SNAPSHOT}"

  if [ "${include_teacher}" = "1" ]; then
    IFS='|' read -r teacher_name teacher_revision < <(
      python3 -c 'import json, sys; c=json.load(open(sys.argv[1])); m=c["models"]["teacher"]; print(m["model_name"] + "|" + m["revision"])' "${config_path}"
    )
    stage_hf_snapshot_to_local "${teacher_name}" "${teacher_revision}"
    export LBD_TEACHER_MODEL_SOURCE="${STAGED_HF_SNAPSHOT}"
  fi
  export LBD_LOCAL_FILES_ONLY=1
}
