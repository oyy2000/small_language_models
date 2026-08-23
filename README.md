# Length Budget Trace Distillation

This folder contains the experimental scaffold for testing whether length-budgeted Qwen teacher traces are better SFT targets for a smaller Qwen student model.

## Capacity-Length Factorial v1

The registered experiment separates requested CoT length from generator
capacity. Qwen2.5-1.5B/3B/7B/14B-Instruct generate three candidates for each
128/256/512-token condition through vLLM. Verified responses become
completion-only SFT targets for a fixed Qwen2.5-1.5B-Instruct student. This is
black-box sequence-level response distillation implemented with SFT, not
logit-level KL distillation.

The 200-problem smoke stage is complete and sealed. It contains 7,200 raw
candidates, 80 common-cohort problems, 78 audited LoRA adapters, 79 evaluation
runs including the base student, and 3,950 per-example predictions. Smoke uses
`GSM8K test[:50]` only to validate the pipeline and does not authorize a paper
conclusion. The full protocol is in
`docs/capacity_length_factorial_protocol.md`.

Launch formal teacher generation and its dependent selection/data build with:

```bash
STAGE=formal bash scripts/5_0_submit_capacity_length_factorial.sh
```

After the formal common-cohort gate passes, launch training, evaluation, and
analysis with the merge job as the upstream dependency:

```bash
STAGE=formal UPSTREAM_JOB_ID=<merge_job_id> \
  bash scripts/6_0_submit_capacity_length_train_eval.sh
```

For the user-approved seed-17 reduced rerun, build a separate dataset manifest
from the immutable selected traces with
`configs/capacity_length_factorial_run_seed17_v1.json`, then launch with
`OUTPUT_ROOT=results/capacity_length_factorial_seed17_v1` and
`CHECKPOINT_ROOT=checkpoints/capacity_length_factorial_seed17_v1/formal`.
This variant trains 26 adapters and evaluates the base model plus those 26
adapters; it is not a three-seed variability estimate.

Seal a completed stage only after the independent evidence audit passes:

```bash
conda run -n sft python scripts/8_1_audit_capacity_length_completion.py \
  --config configs/capacity_length_factorial_v1.json \
  --stage formal \
  --stage-root results/capacity_length_factorial_v1/formal \
  --output-json results/capacity_length_factorial_v1/formal/completion_audit.json
```

The implementation is organized around one real-run pipeline:

- Qwen teacher generation under multiple length budgets.
- GSM8K-style final-answer extraction and verification.
- SFT JSONL export for Qwen student training.
- Reusable logic lives in `src/length_budget_distill/`.
- Runnable entrypoints live in `scripts/`.
- Experiment settings live in `configs/`.
- Outputs should be written under `results/`.

## Project Layout

```text
./
  src/            reusable library code
  scripts/        runnable entry scripts
  configs/        experiment configs
  data/           lightweight local data or processed pointers
  notebooks/      exploratory analysis only
  results/        metrics, traces, logs, artifacts
  figures/        publication-ready plots
  checkpoints/    saved models if needed
  tests/          lightweight sanity tests
  docs/           project notes and usage
```

## Qwen Experiment Template

The templates are prefilled for a small Qwen-series pilot:

- Teacher: `Qwen/Qwen2.5-7B-Instruct`
- Student: `Qwen/Qwen2.5-1.5B-Instruct`
- Dataset: `openai/gsm8k`, config `main`, with GSM8K final-answer extraction

Before any real run, review the choices in:

- `configs/real_length_budget_template.json`
- `configs/student_sft_template.json`

To smoke test the 7B Instruct teacher on a small subset before launching the
full training-set generation, run:

```bash
conda activate fact
CUDA_VISIBLE_DEVICES=0 python3 scripts/1_1_build_length_budget_traces.py \
  --config configs/real_length_budget_template.json \
  --output-dir results/real_length_budget_7b_instruct_smoke \
  --limit 8 \
  --log-every 1

conda activate fact
CUDA_VISIBLE_DEVICES=0 python3 scripts/1_1_build_length_budget_traces.py \
  --config configs/real_length_budget_template.json \
  --output-dir results/real_length_budget_7b_instruct \
  --log-every 1

python3 scripts/1_2_merge_trace_shards.py \
  --input-glob 'results/real_length_budget_7b_instruct_smoke/shard_*.jsonl' \
  --output results/real_length_budget_7b_instruct_smoke/traces_merged.jsonl \
  --sft-output results/real_length_budget_7b_instruct_smoke/sft_merged.jsonl
```

The merged SFT JSONL keeps the original question in `prompt` for training and
stores the full teacher-generation prompt in `teacher_prompt`.

After the smoke test succeeds, rerun the training-set generation with the 7B
Instruct teacher:

```bash
conda activate sft
GPU_IDS=0,1,2,3 LOG_EVERY=10 bash scripts/run_length_budget_4gpu.sh \
  configs/real_length_budget_template.json \
  results/real_length_budget_7b_instruct
```

For this run, per-shard logs are written under
`results/real_length_budget_7b_instruct/logs/`.

The generic sharded launcher command is:

```bash
bash scripts/run_length_budget_4gpu.sh
```

The launcher starts one shard per GPU, waits for all four jobs, and then merges
the shard outputs automatically. Per-shard logs are written under
`results/real_length_budget/logs/`.

To override the GPU list, pass `GPU_IDS`:

```bash
GPU_IDS=0,1,2,3 bash scripts/run_length_budget_4gpu.sh
```

To override config or output directory:

```bash
bash scripts/run_length_budget_4gpu.sh \
  configs/real_length_budget_template.json \
  results/real_length_budget
```

For a standard-prompt versus Chain-of-Draft data comparison, run:

```bash
GPU_IDS=0,1,2,3 bash scripts/1_4_run_prompt_strategy_pair.sh
```

This writes:

```text
results/standard_prompt/sft_merged.jsonl
results/chain_of_draft/sft_merged.jsonl
```

You can also run each condition separately:

```bash
bash scripts/run_length_budget_4gpu.sh \
  configs/standard_prompt_template.json

bash scripts/run_length_budget_4gpu.sh \
  configs/chain_of_draft_template.json
```

Student SFT is exposed through:

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/2_1_train_student_sft.py \
  --config configs/student_sft_template.json
```

The training script expects installed packages such as `transformers`, `datasets`, `trl`, `peft`, and `accelerate`.

The default SFT config uses `data.text_format="prompt_completion"` with
`training.completion_only_loss=true`. This avoids TRL's `assistant_only_loss`
path, because Qwen chat templates may not include the `{% generation %}` marker
needed for assistant-token masks.

## Student SFT Grid Search

Grid search uses a base config plus a grid definition:

- Base config: `configs/student_sft_template.json`
- Grid config: `configs/student_sft_grid_template.json`

Preview all runs without launching training:

```bash
python3 scripts/2_2_grid_search_student_sft.py \
  --base-config configs/student_sft_template.json \
  --grid-config configs/student_sft_grid_template.json \
  --work-dir results/student_sft_grid \
  --dry-run
```

Run the grid on four GPUs:

```bash
python3 scripts/2_2_grid_search_student_sft.py \
  --base-config configs/student_sft_template.json \
  --grid-config configs/student_sft_grid_template.json \
  --work-dir results/student_sft_grid \
  --gpu-ids 0,1,2,3 \
  --skip-existing
```

Each run gets:

```text
results/student_sft_grid/configs/<run>.json
results/student_sft_grid/logs/<run>.log
checkpoints/student_sft_grid/<run>/
```

The default grid sweeps length-specific SFT files, learning rate, and LoRA
rank/alpha. Edit `student_sft_grid_template.json` to add or remove parameters.

## Grid Evaluation

After grid training finishes, evaluate every checkpoint on 50 examples:

```bash
python3 scripts/4_2_eval_grid.py \
  --manifest results/student_sft_grid/manifest.json \
  --config configs/real_length_budget_template.json \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --split test \
  --limit 50 \
  --output-dir results/student_sft_grid/eval \
  --gpu-ids 0,1,2,3
```

Preview eval commands without launching:

```bash
python3 scripts/4_2_eval_grid.py \
  --manifest results/student_sft_grid/manifest.json \
  --config configs/real_length_budget_template.json \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --limit 50 \
  --output-dir results/student_sft_grid/eval \
  --dry-run
```

The aggregate reports are:

```text
results/student_sft_grid/eval/grid_eval_summary.json
results/student_sft_grid/eval/grid_eval_summary.csv
```

## SFT Before/After Evaluation

Evaluate the base student before SFT:

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/4_1_eval_model.py \
  --config configs/real_length_budget_template.json \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --split test \
  --limit 200 \
  --output-jsonl results/eval/qwen_base_gsm8k_test.jsonl \
  --summary-json results/eval/qwen_base_gsm8k_test_summary.json
```

After SFT, evaluate the same base model with the saved LoRA adapter:

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/4_1_eval_model.py \
  --config configs/real_length_budget_template.json \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-path checkpoints/student_sft_pilot \
  --split test \
  --limit 200 \
  --output-jsonl results/eval/qwen_sft_gsm8k_test.jsonl \
  --summary-json results/eval/qwen_sft_gsm8k_test_summary.json
```

The before/after comparison is the accuracy difference between the two summary
files. Both commands use the same split, prompt, answer extractor, and verifier.

## How Length-Budgeted Training Data Is Generated

The different-length training examples are generated by sweeping `length_budgets` in `configs/real_length_budget_template.json`.

This follows the prompt-control setup: the teacher is asked to solve in `<= k`
solution tokens while `k` is swept across bins. The decoder still receives a
larger safety limit through `teacher.generation.max_new_tokens`, so short
generations are not cut off before the final `Answer:` line.

For every problem assigned to the current shard, `1_1_build_length_budget_traces.py` loops over each budget entry:

```json
{
  "name": "small",
  "max_solution_tokens": 128,
  "style_hint": "Use compressed equations and only essential words, but always include the final Answer line."
}
```

For each `(problem, budget)` pair, the script:

1. Builds a teacher prompt that includes the target token budget `k` and style hint.
2. Calls the Qwen teacher backend to generate one visible solution trace.
3. Extracts the final answer and verifies it against the dataset answer.
4. Counts solution tokens with the configured Qwen tokenizer.
5. Writes one raw trace with `budget_name`, `max_solution_tokens`, `solution`, and correctness metadata.
6. Writes one SFT record when the trace is correct, because `keep_only_correct_for_sft` is enabled.

So one original math problem becomes up to three supervised examples:

```text
problem_001:small   -> compact solution target
problem_001:medium  -> medium solution target
problem_001:large   -> detailed solution target
```

After all shards finish, `1_2_merge_trace_shards.py` merges the shard files and writes `sft_merged.jsonl`. Each SFT row keeps its length label in `metadata.budget_name`, so downstream training can use all budgets together or filter by `small`, `medium`, or `large` for separate runs.

To materialize separate SFT files by length, filter on `metadata.budget_name`:

```bash
python3 -c 'import json, pathlib
src = pathlib.Path("results/real_length_budget/sft_merged.jsonl")
out_dir = pathlib.Path("results/real_length_budget")
for name in ("small", "medium", "large"):
    with src.open() as fin, (out_dir / f"sft_{name}.jsonl").open("w") as fout:
        for line in fin:
            row = json.loads(line)
            if row.get("metadata", {}).get("budget_name") == name:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
'
```

Then point `data.train_path` in `configs/student_sft_template.json` to one of:

```text
results/real_length_budget/sft_small.jsonl
results/real_length_budget/sft_medium.jsonl
results/real_length_budget/sft_large.jsonl
```


 conda activate sft
  python3 scripts/1_3_plot_sft_dataset_stats.py \
    --dataset cod=results/chain_of_draft/sft_merged.jsonl \
    --dataset cot=results/standard_prompt/sft_merged.jsonl \
    --output-json data/cot_vs_codstats.json \
    --output-csv data/cot_vs_cod_stats.csv \
    --figure figures/cot_vs_cod_stats.png

## Capacity-by-Length Factorial Experiment

The registered paper experiment tests whether downstream gains come from CoT
length alone or from an interaction between generator capacity and CoT length.
It uses Qwen2.5-1.5B/3B/7B/14B-Instruct generators, 128/256/512-token budgets,
three candidates per problem-condition, and a fixed Qwen2.5-1.5B-Instruct
student. The 1.5B generator is a self-distillation control.

This pipeline is offline black-box sequence-level response distillation: vLLM
runs inference for the generator, and the selected verified responses become
completion-only SFT targets. It is not teacher-logit or KL distillation.

Run the four-node 200-problem smoke generation and dependent audit/data build:

```bash
STAGE=smoke bash scripts/5_0_submit_capacity_length_factorial.sh
```

Preview the exact Slurm commands without submission:

```bash
DRY_RUN=1 STAGE=smoke bash scripts/5_0_submit_capacity_length_factorial.sh
```

After the smoke audit passes, submit the same command with `STAGE=formal` for
the 2,000-problem, 72,000-candidate matrix. Training consumes
`<stage>/sft_data/dataset_manifest.json`; it creates 36 equal-example runs, 36
equal-token robustness runs, and six calibration runs. Formal evaluation is
locked to GSM8K `test[50:1319]`.

```bash
STAGE=formal UPSTREAM_JOB_ID=<merge_job_id> \
  bash scripts/6_0_submit_capacity_length_train_eval.sh
```

Large artifacts live under
`results/capacity_length_factorial_v1`, backed by BeeGFS. Each phase writes a
manifest and completion marker with SHA256 evidence; partial shards and queued
jobs are not results. Final analysis reports teacher pass@3/candidate quality
separately from downstream student accuracy and writes publication-oriented
capacity-by-length figures.
