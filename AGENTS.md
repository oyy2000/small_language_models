# Project Instructions

## Global Rules

- Do not use emojis in project documentation, logs, or generated reports.
- Maintain a professional, academic, and clean writing style.
- Preserve existing user changes and historical experimental artifacts.

## Project Layout

- Put reusable Python logic in `src/length_budget_distill/`.
- Put runnable phase-first entrypoints in `scripts/`.
- Put experiment settings in `configs/` rather than hard-coding them.
- Store lightweight metadata in `data/`, metrics and artifacts in `results/`, publication figures in `figures/`, and model adapters in `checkpoints/`.
- Keep notebooks exploratory and migrate reusable logic into `src/` or `scripts/`.
- Prefer descriptive phase-first script names such as `5_1_generate_capacity_length_traces.py`.

## Script and Workflow Conventions

- Put reusable logic in `src/`, expose it through a small runnable script in `scripts/`, keep settings in `configs/`, write artifacts to `results/`, and place publication plots in `figures/` or the experiment's analysis artifact directory.
- Before implementing a new utility or figure, search existing `src/`, `scripts/`, and figure-generation code for reusable logic and established visual conventions. Prefer extracting or extending shared code over creating a parallel implementation; document the reason when an intentional visual or implementation divergence is necessary.
- Runnable scripts should do one clear job, expose explicit arguments and inputs/outputs, minimize hidden side effects, and log major settings.
- Use phase-first lowercase names in the form `{phase}_{subphase}_{short_description}.py`; do not use numeric-only or vague `final_v2` names.
- Avoid duplicating logic across scripts.

## Result Presentation

- Prefer figures and visual summaries for experimental results: line plots for trends, bars for method comparisons, scatter plots for correlations, pipeline flows, and ablation figures.
- Use tables only when exact values or compact comparisons are necessary.

## Experiment Evidence

- Keep exploratory, smoke-test, and formal evidence in separate directories.
- Do not treat queued jobs, partial shards, or stale summaries as completed experiments.
- A formal result requires complete shard manifests, duplicate/missing-record audits, config and input hashes, complete checkpoints, per-example predictions, aggregate statistics, and a completion marker.
- Do not reuse the historical single-candidate 7B traces as formal evidence for the capacity-by-length factorial experiment.
- Limit the current paper claim to GSM8K unless a separate OOD protocol is approved.

## Compute Resources

- Approved machines are C30, C31, C32, and C49.
- For interactive access to C30, C31, or C32, use `salloc` to obtain the allocation instead of using a bare `srun` request. Use the node's registered partition, for example `salloc --partition=a6000 --nodelist=c31 --nodes=1 --ntasks=1 --cpus-per-task=16 --time=<duration>` for C30/C31 and `--partition=a5000ada --nodelist=c32` for C32.
- After an interactive `salloc` is granted, run node checks or manual commands inside it with `srun --jobid="${SLURM_JOB_ID}" --overlap ...`. Do not rely on direct SSH without an active allocation.
- Use `sbatch` for queued, unattended experiment jobs and dependency-ordered DAGs. Do not use a pending `salloc` request as the experiment queue or as a replacement for `sbatch`.
- Keep an interactive `salloc` allocation only while the interactive session is active; this rule does not replace the separate requirement to keep user-owned `grabgpu` allocations open.
- Keep `grabgpu` allocations open unless the user explicitly asks to close them.
- On C49, `/home/youyang7/bin/gg` processes launched by the user's active `grabgpu` allocation are GPU keepalive/occupancy processes, not competing workloads. Do not require low utilization or near-zero used memory on C49: a GPU is available whenever its remaining free memory is sufficient for the workload's expected peak allocation plus a safety margin. Ignore `gg` utilization when deciding availability and account for its memory only by subtracting it from the available capacity.
- Do not terminate the C49 `gg` processes or close their allocation. Run project workloads inside the existing C49 allocation with `srun --jobid=<allocation> --overlap`; continue to identify non-`gg` GPU processes before launch and include their memory in the free-capacity calculation without interfering with them.
- Inspect `nvidia-smi` before every launch. Never terminate or interfere with another process or allocation.
- Slurm `Gres=(null)` and `--exclusive` do not prove GPU isolation on these nodes; verify physical GPU occupancy directly.
- On oversubscribed partitions, use the shared stable-idle-GPU gate in `scripts/slurm/_gpu_idle_gate.sh`; CPU sharing is permitted only after selecting GPUs with low memory use and utilization across repeated checks.
- Prefer one process per available GPU, disjoint problem shards, per-shard outputs, and an audited merge.
- Use model/tensor parallelism only when a model does not fit safely on one GPU.
- Use `srun --jobid=<allocation> --overlap` inside an existing `salloc` or `grabgpu` allocation rather than cancelling it.
- Do not request Slurm memory from the node's misleading `RealMemory` field; enforce process memory only when the workload requires it.
- Put large checkpoints and formal experiment artifacts on BeeGFS while retaining stable project paths.
- For factorial SFT/evaluation, keep Hugging Face dataset caches, temporary files, and Trainer intermediate checkpoints on node-local `/var/tmp`; publish only final hash-verified LoRA files and registered evidence to BeeGFS with bounded retries.

## Capacity-Length Factorial Protocol

- Formal generators are Qwen2.5-1.5B/3B/7B/14B-Instruct, with 128/256/512-token solution budgets and three candidates per problem-condition.
- Call the 1.5B generator cell a self-distillation control. The pipeline is black-box sequence-level response distillation implemented with SFT, not logit-level KD.
- Use the fixed Qwen2.5-1.5B-Instruct student and fixed LoRA hyperparameters across all factorial conditions.
- Use `configs/capacity_length_factorial_sft_v1.json` for the registered batch-size-4, gradient-accumulation-1 overlay; it is parent-hash-bound so training-only corrections do not invalidate immutable teacher traces.
- Use `test[:50]` only for smoke checks and `test[50:1319]` as the locked formal GSM8K evaluation cohort.
- Primary training is equal-example on the 12-condition common problem intersection. Equal-supervision-token training is a robustness analysis.
- A factorial adapter is complete only when its marker verifies the training-data hash, run-config hash, training and launcher source hashes, and both final LoRA file hashes.
- Use the parent-hash-bound `configs/capacity_length_factorial_eval_v1.json` overlay for batched greedy evaluation; do not silently change the locked split or decoding parameters.
- Preserve the original three-seed `capacity_length_factorial_v1` artifacts as historical evidence. The user-approved reduced rerun uses only seed 17 through `configs/capacity_length_factorial_run_seed17_v1.json`, reuses the immutable parent teacher traces, writes to a separate `capacity_length_factorial_seed17_v1` result/checkpoint root, and must be labeled as a revised single-seed protocol that does not estimate training-seed variability.
