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
