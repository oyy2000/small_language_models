# Short–medium–long 轨迹蒸馏研究

更新日期：2026-08-29

## 结论摘要

这个方向值得扩大，但下一步不应直接把全部模型和全部条件做成一个大网格。当前结果最有价值的地方，是在同一道题、同一个无长度约束 prompt、同一个 teacher 候选池中，先过滤答案正确的轨迹，再选择相对 short、medium、long 的完整解答训练同一个 student。它比直接给 teacher 加 128/256/512 token 提示更好地控制了题目与 prompt 差异。

当前 4 teachers x 3 ranks x 3 seeds 主矩阵已经完成。在四个 teacher 下，三-seed平均准确率都呈 `short > medium > long`，student 输出长度都呈 `short < medium < long`。1.5B self-distillation control 和 3B teacher 的 short-long accuracy effect 分别为 `+4.18 pp` 和 `+4.96 pp`，通过 12 项比较的 Holm 校正；7B 和 14B 的效应分别为 `+3.10 pp` 和 `+1.55 pp`，没有通过校正。六项 teacher-by-rank interaction 均未通过 Holm 校正，因此当前证据支持稳定的长度传递和部分 teacher 内 short-long 优势，但尚未证明 teacher capacity 会调节该效应。

文献中已经有人使用“多次采样后取最短正确响应”的核心统计量。最直接的近邻是 [ShorterBetter](https://arxiv.org/abs/2504.21370)，其 Sample Optimal Length 就是多次 rollout 中最短正确响应的长度，并把它用于 GRPO 奖励。因此，本项目不能声称首创 shortest-correct selection。更稳健的论文定位是：

> 系统研究 answer-blind teacher 候选分布中的 within-question relative length rank，如何在固定问题支持集、固定 student 和固定 SFT 配方下影响小模型的准确率、推理长度与截断失败，并进一步测量 teacher capacity、student capacity、sampling density 和监督 token 总量的交互。

完整文档：

- [文献调研与论文定位](literature_review.md)
- [大规模实验设计](large_scale_experiment_plan.md)
- [4 teachers x 3 ranks x 3 seeds 主矩阵结果](main_matrix_results.md)

## Phase A 证据锚点

| 条件 | 三-seed平均准确率 | 平均输出 token | 训练轨迹平均 token |
|---|---:|---:|---:|
| Base 1.5B | 67.14% | 242.6 | 不适用 |
| Short-ranked SFT | 70.32% | 263.9 | 194.4 |
| Medium-ranked SFT | 69.58% | 295.1 | 242.5 |
| Long-ranked SFT | 67.59% | 331.1 | 318.0 |

核心比较为 short 对 long：三 seed 平均 `+2.73 pp`，crossed seed/problem bootstrap 95% CI `[+0.60,+4.89] pp`，Holm-adjusted bootstrap `p=0.0462`。三个 seed 的差异分别为 `+2.29/+2.60/+3.31 pp`。这是对已观察 GSM8K `test[50:1319]` 的比较性复现，独立评测问题仍是 1,269 道，不是 3,807 道。

## Phase C 主矩阵证据锚点

| Teacher | Short accuracy | Medium accuracy | Long accuracy | Short-long effect | Holm p |
|---|---:|---:|---:|---:|---:|
| 1.5B self-distillation control | 72.29% | 69.27% | 68.11% | +4.18 pp | <0.0001 |
| 3B | 72.03% | 69.77% | 67.06% | +4.96 pp | 0.0044 |
| 7B | 70.42% | 69.37% | 67.32% | +3.10 pp | 0.0828 |
| 14B | 70.63% | 70.29% | 69.08% | +1.55 pp | 0.8060 |

主矩阵已完成 36 个 adapter、37 个评测模型和 46,953 条逐题 prediction。最终审计通过；详细统计、限制和恢复记录见[主矩阵结果报告](main_matrix_results.md)。

证据路径：

- [三-seed正式报告](../../results/capacity_length_ranked_sampling_7b_multiseed_v1/formal/analysis/experiment_report.md)
- [三-seed完成审计](../../results/capacity_length_ranked_sampling_7b_multiseed_v1/formal/completion_audit.json)
- [三-seed完成标记](../../results/capacity_length_ranked_sampling_7b_multiseed_v1/MULTISEED_COMPLETE)
- [生成审计](../../results/capacity_length_ranked_sampling_7b_v1/formal/datasets/generation_audit.json)
- [聚合配对比较](../../results/capacity_length_ranked_sampling_7b_multiseed_v1/formal/analysis/aggregate_paired_contrasts.csv)
- [主图 PNG](../../figures/capacity_length_ranked_sampling_7b_multiseed_v1/formal/ranked_multiseed_accuracy_and_output_length.png)
- [主图 PDF](../../figures/capacity_length_ranked_sampling_7b_multiseed_v1/formal/ranked_multiseed_accuracy_and_output_length.pdf)
- [主矩阵正式报告](../../results/capacity_length_ranked_sampling_multiteacher_v1/formal/analysis/experiment_report.md)
- [主矩阵完成审计](../../results/capacity_length_ranked_sampling_multiteacher_v1/formal/completion_audit.json)
- [主矩阵完成标记](../../results/capacity_length_ranked_sampling_multiteacher_v1/MATRIX_COMPLETE)
- [主矩阵主图 PNG](../../figures/capacity_length_ranked_sampling_multiteacher_v1/formal/teacher_capacity_by_rank_accuracy_and_output_length.png)
- [主矩阵主图 PDF](../../figures/capacity_length_ranked_sampling_multiteacher_v1/formal/teacher_capacity_by_rank_accuracy_and_output_length.pdf)

## 推荐执行顺序

1. Phase A 已完成：新增 seed 42 和 73 的 6 个 adapter，并与 seed 17 合并完成三-seed评测、统计和审计。
2. Phase C 主矩阵已完成：固定 `K=16`，在 1.5B student 上完成 `teacher={1.5B,3B,7B,14B} x rank={short,medium,long} x seed={17,42,73}` 的全部 36 个 adapter、锁定评测、统计、图表与完成审计。四个 teacher 的平均准确率都为 short 最高，但只有 1.5B control 和 3B 的 short-long contrast 通过 12 项 Holm 校正；六项 teacher-by-rank interaction 均未通过校正。
3. `K_max=32` 与嵌套 `K={4,8,16,32}` 的 Phase B 数据审计保留为后续 sampling-density robustness，不得与当前 `K=16` 主矩阵混报。
4. 当前 teacher-by-rank interaction 尚未通过预设 gate，因此暂不直接扩完整 student-capacity 矩阵；如要检验该交互，应另行预注册更高 power 的复现。
5. 在少数关键端点补 equal-supervision-token、随机正确候选、prompt-constrained short 和 paired rewrite 控制，区分长度、内容策略和训练曝光量。

项目内 GSM8K official test 已全部被观察，[paired-rewrite confirmatory manifest](../../results/capacity_length_paired_rewrite_7b_pilot_v1/pilot/eval/confirmatory/eval_manifest_confirmatory_shard_00_of_01.json) 也表明 `train[3000:7473]` 已被用于评测。因此后续 GSM8K 结果应标为锁定比较性复现，不应称为新的 untouched confirmation。任何新的 OOD 确认性声明都需要另立并审批协议；当前文档没有扩大现有 GSM8K 论文声明范围。
