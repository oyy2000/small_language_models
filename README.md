# Length Budget Trace Distillation

This folder contains the experimental scaffold for testing whether length-budgeted Qwen teacher traces are better SFT targets for a smaller Qwen student model.

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

- Teacher: `Qwen/Qwen2.5-Math-7B-Instruct`
- Student: `Qwen/Qwen2.5-1.5B-Instruct`
- Dataset: `openai/gsm8k`, config `main`, with GSM8K final-answer extraction

Before any real run, review the choices in:

- `configs/real_length_budget_template.json`
- `configs/student_sft_template.json`

Then run the generation script with sharding. On a 4-GPU machine with GPU IDs
`0,1,2,3`, use the launcher:

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
  --gpu-ids 0,1,2,3
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
