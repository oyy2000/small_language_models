# TODO

## Capacity-Length Factorial v1

- [x] Freeze the 1.5B/3B/7B/14B generator matrix and 128/256/512-token budgets.
- [x] Add the user-approved seed-17 reduced rerun as a parent-hash-bound, separately labeled protocol variant.
- [x] Implement three-candidate vLLM generation with deterministic per-request seeds.
- [x] Implement shortest-correct, budget-compliant selection and common-problem gates.
- [x] Implement equal-example and equal-supervision-token SFT datasets.
- [x] Implement fixed-config, three-seed training and locked GSM8K evaluation launchers.
- [x] Implement clustered interaction analysis, paired bootstrap, Holm correction, and figures.
- [x] Run and audit the 200-problem smoke matrix: 7,200 candidates, 80 common problems, 78 adapters, 79 evaluations, and zero completion-audit errors.
- [ ] Run the 2,000-problem formal matrix with the smoke-validated environment (completed generation: 7B `275891` and 3B recovery `275912`; active C31 serial recovery: 14B `276000` then 1.5B `276001`; merge `275899`; downstream `275900`-`275910`).
- [ ] Complete 78 registered SFT runs and 79 locked evaluations including the base model.
- [ ] Produce the final GSM8K-only experiment report and completion marker.

## Evidence Gates

- Smoke selection requires at least 50 problems with eligible traces in all 12 conditions.
- Formal selection requires at least 500 common problems and exactly 72,000 raw candidates.
- Do not promote old single-candidate 7B traces or the previously tuned first 50 test examples to formal evidence.
- Do not make OOD or MATH claims from this registered GSM8K-only protocol.
- A stage is sealed only when `scripts/8_1_audit_capacity_length_completion.py` passes and writes the stage completion marker.

## Multi-Benchmark, Multi-Teacher KD Pilot v1

- [x] Freeze the four-teacher by three-length, GSM8K+MATH equal-token design as an independent exploratory root.
- [x] Reuse the completed 7B MATH trajectories and submit missing 1.5B/3B/14B MATH generation (`276312`, completed `276316`, `276329`) plus audited build (`276317`).
- [x] Submit matched SFT, inherited-hyperparameter logit-KD, 25-model evaluation, analysis, and completion-audit DAG (`276318`-`276328`).
- [ ] Complete and audit all three missing-teacher raw trajectory matrices.
- [ ] Complete 12 matched SFT adapters and 12 matched-teacher logit-KD adapters.
- [ ] Complete 75 model-by-benchmark evaluation artifacts and publication figures.
- [ ] Require `results/capacity_length_multibench_multiteacher_kd_pilot_v1/PILOT_COMPLETE` before reporting results as completed exploratory evidence.

## 7B Teacher-Prompt Logit-KD Ablation

- [x] Register the dual-context teacher-forced protocol on the immutable 7B equal-token subsets.
- [x] Add strict completion-token alignment, dual-context GPU smoke, and context-bound evidence.
- [x] Add a two-percentage-point validation compliance gate with no accuracy-first fallback.
- [ ] Complete the 27-adapter validation grid and select one shared feasible alpha/temperature pair.
- [ ] Complete three diagnostic adapters, 3,807 predictions, matched-logit evidence, figures, and independent audit.
- [ ] Require the new root `FORMAL_COMPLETE` before treating the ablation as completed exploratory evidence.
