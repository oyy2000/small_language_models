# Capacity-Length Factorial v1

## Research Question

The registered experiment tests whether a downstream student benefits from
short CoTs because of length alone or because the useful length depends on the
capacity of the model that generated the trace. "Shorter CoTs are better" is a
directional hypothesis, not an assumed result.

The primary factors are generator capacity (Qwen2.5 1.5B, 3B, 7B, and 14B
Instruct) and requested solution length (128, 256, and 512 tokens). The student
is fixed to Qwen2.5-1.5B-Instruct. The 1.5B generator is a self-distillation
control.

## Method Classification

Qwen responses are generated offline through vLLM, verified, and converted to
completion-only SFT examples. This is black-box sequence-level response
distillation implemented through SFT. It is not logit-level knowledge
distillation because the pipeline does not consume teacher logits or optimize a
teacher-student KL objective.

Using an instruction-tuned teacher to generate synthetic targets for a smaller
instruction-tuned student is consistent with common production response
distillation workflows. The experiment nevertheless separates generator
capacity from trace length rather than assuming the largest teacher is always
best.

## Registered Data Protocol

- Source: first 2,000 GSM8K train examples.
- Conditions: four generator sizes by three length budgets.
- Sampling: three candidates per problem-condition, temperature 0.7, top-p
  0.95, deterministic identity-derived seeds.
- Selection: shortest candidate that is answer-correct and no longer than the
  requested budget.
- GSM8K verification extracts the first numeric value from the explicit final
  answer segment, so semantically equivalent forms such as `72`, `72 clips`,
  and `$1,250.00` are normalized consistently. Merge re-verifies immutable raw
  traces and audits any stored-label mismatch.
- Primary cohort: problem IDs with an eligible selected trace in all 12
  conditions.
- Smoke gate: at least 50 common problems from the 200-problem smoke matrix.
- Formal gate: at least 500 common problems and exactly 72,000 raw candidates.

The equal-example experiment uses the identical common problem cohort in every
condition. The equal-token robustness experiment deterministically subsamples
whole traces to the smallest condition-level supervision-token total.

## Training and Evaluation

All factorial conditions use the same LoRA configuration: rank 4, alpha 16,
dropout 0.05, all linear modules, learning rate 2e-5, one epoch, per-device
batch size 4, no gradient accumulation, completion-only loss, and seeds
17/42/73. Gold-rationale and answer-only SFT are calibration baselines. The
SFT-specific settings live in `configs/capacity_length_factorial_sft_v1.json`,
which is bound to the immutable generation protocol hash. Accumulation is one
so Transformers 4.48.3 consumes a final partial batch instead of ending an
epoch with an unapplied tail accumulation.

SFT construction and evaluation call the same student math-prompt builder. In
the installed TRL 0.9.6 environment, completion-only loss is enforced with an
explicit Qwen assistant-boundary data collator because `SFTConfig` does not
natively expose `completion_only_loss`; the boundary is validated against the
tokenizer chat template before training begins.

Trainer caches and intermediate checkpoints are written under node-local
`/var/tmp`. After a successful run, only `adapter_config.json` and
`adapter_model.safetensors` are copied to the BeeGFS-backed canonical checkpoint
directory with post-copy SHA256 verification and bounded retry. The completion
marker binds those file hashes to the training data, run config, training
implementation, and launcher implementation.

Greedy student evaluation is decoded in batches of 32 using left padding.
The batching setting lives in `configs/capacity_length_factorial_eval_v1.json`
and is bound to the immutable parent protocol hash; it changes throughput, not
the registered temperature-zero decoding rule.

GSM8K `test[:50]` is limited to smoke checks because it was used in prior
exploration. The formal cohort is locked to `test[50:1319]`, yielding 1,269
identical held-out problems for every run. No OOD or MATH claim is authorized by
this protocol.

## Evidence Contract

Formal completion requires shard manifests, exact candidate cardinality,
duplicate and candidate-index audits, common-cohort evidence, config and source
hashes, 78 complete adapters, the base plus 78 adapter evaluations, per-example
predictions, registered interaction tests, figures, and completion markers.
Queued jobs and partial shards are operational status only.

The primary model is a clustered binomial regression with categorical generator
capacity, categorical length budget, their interaction, and training seed.
Planned paired contrasts use problem-cluster bootstrap confidence intervals and
Holm correction. Equal-token results are a required robustness check for the
equal-example conclusion.

Teacher-trace quality is reported separately as pass@3 and raw candidate
correctness for every capacity-by-length cell. These descriptive generation
metrics answer whether short trajectories from smaller or larger generators are
more often usable; they are not substituted for downstream student accuracy.

The final independent audit is exposed by
`scripts/8_1_audit_capacity_length_completion.py`. It re-hashes the raw shards,
SFT datasets, adapters, predictions, summaries, and analysis artifacts; checks
the exact registered run identities and cardinalities; and writes a stage-level
completion marker only when no errors remain. Smoke analysis is always labeled
`smoke_only_no_scientific_conclusion`, even when its small validation cohort
shows an apparent statistical pattern.
